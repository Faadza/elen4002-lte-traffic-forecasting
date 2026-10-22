"""Chronological early stopping followed by a fresh full-prefix LSTM refit.

Only the inner tail chooses the epoch count. The returned model starts afresh
with the same seed and learns from all permitted fitting hours for that count.
"""

from dataclasses import asdict, dataclass
import hashlib
from numbers import Real
from time import perf_counter_ns

import numpy as np
import torch

from evaluation.metrics import finalize_predictions, mae
from forecasting.data import fit_scaler, lstm_inputs, make_examples, scaled_targets
from models.lstm import build_model, predict_lstm, train_epoch


@dataclass(frozen=True)
class LSTMFit:
    model: torch.nn.Module
    scaler: object
    metadata: dict
    selection_model: torch.nn.Module | None = None


@dataclass(frozen=True)
class EarlyStopping:
    best_epoch: int = 0
    best_score: float = float("inf")
    epochs_since_improvement: int = 0
    epochs_seen: int = 0
    should_stop: bool = False


def _positive_integer(value, name, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1 or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the supported positive range")
    return int(value)


def update_early_stopping(state, epoch, score, patience):
    """Strict improvement resets patience; exact ties retain the earliest best."""
    patience = _positive_integer(patience, "patience")
    epoch = _positive_integer(epoch, "epoch")
    if not isinstance(state, EarlyStopping) or epoch != state.epochs_seen + 1 or state.should_stop:
        raise ValueError("Early-stopping epochs must advance in order before stopping")
    if isinstance(score, (bool, np.bool_)) or not isinstance(score, Real) or not np.isfinite(score) or score < 0:
        raise ValueError("The tail MAE must be finite and nonnegative")
    improved = score < state.best_score
    stale = 0 if improved else state.epochs_since_improvement + 1
    return EarlyStopping(epoch if improved else state.best_epoch,
                         float(score) if improved else state.best_score,
                         stale, epoch, stale >= patience)


def state_fingerprint(model):
    """Hash named CPU parameter arrays, independent of a pickle container."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _settings(settings, hidden_units, dropout, seed):
    fixed = {
        "layers": 1,
        "hidden_units": [16, 32],
        "output_dropout": [0.0, 0.2],
        "head": "last hidden state -> dropout -> concatenate 4 calendar values -> linear scalar",
        "stateless_windows": True,
        "shuffle_windows": False,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "training_loss": "MSE in standardised units",
        "seeds_all_formal_periods": [2026, 2027, 2028],
        "inner_stopping_metric": "MAE after inverse transform and final lower bound on eligible observed tail targets",
        "improvement": "strictly lower score; earliest best epoch on tie",
        "refit": "reinitialise same seed; fit scaler/model on full permitted prefix for best inner epoch count",
        "confirmation_final_epochs": "ceil(median(selected-architecture best epochs across development cells/folds/seeds))",
        "confirmation_final_early_stopping": False,
        "prediction_ensemble": False,
        "remaining_library_defaults": "record exact installed-version defaults during environment setup",
    }
    limits = {"maximum_epochs": 100, "patience_epochs": 10, "inner_stopping_tail_hours": 336}
    if not isinstance(settings, dict) or set(settings) != set(fixed) | set(limits):
        raise ValueError("LSTM settings must contain exactly the declared configuration keys")
    for name, expected in fixed.items():
        value = settings[name]
        if isinstance(expected, bool):
            valid = value is expected
        elif name in ("layers", "batch_size"):
            valid = not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)) and value == expected
        elif name == "learning_rate":
            valid = not isinstance(value, (bool, np.bool_)) and isinstance(value, Real) and np.isfinite(value) and value == expected
        else:
            valid = value == expected
        if not valid:
            raise ValueError(f"LSTM setting {name} differs from the approved protocol")
    budget = {name: _positive_integer(settings[name], name, maximum)
              for name, maximum in limits.items()}
    if _positive_integer(hidden_units, "hidden_units") not in fixed["hidden_units"]:
        raise ValueError("hidden_units must be in the declared grid")
    if isinstance(dropout, (bool, np.bool_)) or not isinstance(dropout, Real) or dropout not in fixed["output_dropout"]:
        raise ValueError("dropout must be in the declared grid")
    if _positive_integer(seed, "seed") not in fixed["seeds_all_formal_periods"]:
        raise ValueError("seed must be in the declared seed set")
    return budget


def _prefix(values, observed, timestamps, fit_end):
    fit_end = _positive_integer(fit_end, "fit_end")
    if not (len(values) == len(observed) == len(timestamps)) or fit_end > len(values):
        raise ValueError("The fitting interval must lie within matching hourly arrays")
    # Slice before converting, validating or building examples: later data stay out.
    return values[:fit_end], observed[:fit_end], timestamps[:fit_end]


def _training_arrays(examples, scaler):
    if len(examples.targets) == 0:
        raise ValueError("The fitting segment has no eligible observed targets")
    sequence, calendar = lstm_inputs(examples, scaler)
    with np.errstate(over="ignore", invalid="ignore"):
        targets = scaled_targets(examples, scaler).astype(np.float32)
    if not np.isfinite(targets).all():
        raise ValueError("LSTM float32 target conversion produced a nonfinite value")
    return sequence, calendar, targets


def _new_model(hidden_units, dropout, seed, settings):
    model = build_model(hidden_units=hidden_units, dropout=dropout, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    # Every phase receives a new model and an optimizer with no accumulated moments.
    if optimizer.state:
        raise AssertionError("A fresh Adam optimizer must have empty state")
    return model, optimizer


def _fit_epochs(model, optimizer, arrays, epochs, batch_size, progress):
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, optimizer, *arrays, batch_size=batch_size)
        history.append({"epoch": epoch, "train_mse": float(loss)})
        if progress is not None:
            progress({"phase": "refit", **history[-1]})
    return history


def fit_development(values, observed, timestamps, fit_end, *, hidden_units,
                    dropout, seed, settings, progress=None, record_phase_times=False):
    """Choose epochs inside the prefix, then return a freshly refitted model.

    Smaller epoch/patience/tail budgets support invented unit-test fixtures.
    The experiment runner must pass the unchanged committed configuration.
    Optional phase times include preparation and existing checks/callbacks.
    Shared argument validation and final metadata construction remain outside.
    """
    if type(record_phase_times) is not bool:
        raise ValueError("record_phase_times must be a boolean")
    budget = _settings(settings, hidden_units, dropout, seed)
    past, flags, clock = _prefix(values, observed, timestamps, fit_end)
    inner_end = fit_end - budget["inner_stopping_tail_hours"]
    if inner_end <= 168:
        raise ValueError("The inner fitting segment needs eligible history before its tail")
    if record_phase_times:
        selection_start = perf_counter_ns()
    scaler = fit_scaler(past[:inner_end], flags[:inner_end], inner_end)
    inner = make_examples(past[:inner_end], flags[:inner_end], clock[:inner_end], 0, inner_end)
    tail = make_examples(past, flags, clock, inner_end, fit_end)
    arrays = _training_arrays(inner, scaler)
    if len(tail.targets) == 0:
        raise ValueError("The inner stopping tail has no eligible observed targets")
    tail_sequence, tail_calendar = lstm_inputs(tail, scaler)
    model, optimizer = _new_model(hidden_units, dropout, seed, settings)
    inner_initial = state_fingerprint(model)
    optimizer_defaults = dict(optimizer.defaults)
    state, history = EarlyStopping(), []
    for epoch in range(1, budget["maximum_epochs"] + 1):
        loss = train_epoch(model, optimizer, *arrays, batch_size=settings["batch_size"])
        tail_z = predict_lstm(model, tail_sequence, tail_calendar, batch_size=settings["batch_size"])
        final, _ = finalize_predictions(scaler.inverse(tail_z))
        score = mae(tail.actual, final)
        state = update_early_stopping(state, epoch, score, budget["patience_epochs"])
        row = {"epoch": epoch, "train_mse": float(loss), "tail_mae": float(score),
               "best_epoch": state.best_epoch, "epochs_since_improvement": state.epochs_since_improvement}
        history.append(row)
        if progress is not None:
            progress({"phase": "selection", **row})
        if state.should_stop:
            break

    if record_phase_times:
        selection_end = perf_counter_ns()
        refit_start = perf_counter_ns()
    full_scaler = fit_scaler(past, flags, fit_end)
    full = make_examples(past, flags, clock, 0, fit_end)
    full_arrays = _training_arrays(full, full_scaler)
    refit_model, refit_optimizer = _new_model(hidden_units, dropout, seed, settings)
    refit_initial = state_fingerprint(refit_model)
    if inner_initial != refit_initial:
        raise AssertionError("Same-seed reinitialisation did not reproduce initial parameters")
    refit_history = _fit_epochs(refit_model, refit_optimizer, full_arrays,
                                state.best_epoch, settings["batch_size"], progress)
    if record_phase_times:
        refit_end = perf_counter_ns()
    metadata = {
        "mode": "development_inner_selection_then_fresh_refit",
        "hidden_units": int(hidden_units), "dropout": float(dropout), "seed": int(seed),
        "inner_fit_interval": [0, inner_end], "inner_tail_interval": [inner_end, fit_end],
        "refit_interval": [0, fit_end], "inner_fit_count": len(inner.targets),
        "inner_tail_count": len(tail.targets), "refit_count": len(full.targets),
        "inner_scaler": asdict(scaler), "refit_scaler": asdict(full_scaler),
        "selected_epochs": state.best_epoch, "best_tail_mae": state.best_score,
        "selection_epochs_run": state.epochs_seen, "stopping_history": history,
        "refit_history": refit_history,
        "stop_reason": "patience" if state.should_stop else "maximum_epochs",
        "maximum_epochs_reached": state.epochs_seen == budget["maximum_epochs"],
        "initial_state_sha256": {"inner_selection": inner_initial, "refit": refit_initial},
        "same_seed_initialisation_verified": True,
        "optimizer_defaults": optimizer_defaults, "fresh_optimizer_each_phase": True,
        "fresh_optimizer_state_empty": {"inner_selection": True, "refit": True},
        "component_fit_counts": {"inner_selection": 1, "full_prefix_refit": 1},
        "effective_budgets": budget,
    }
    if record_phase_times:
        if not selection_start < selection_end <= refit_start < refit_end:
            raise RuntimeError("LSTM phase timings require an advancing monotonic clock")
        metadata["phase_timings"] = {
            "clock": "perf_counter_ns",
            "selection_seconds": (selection_end - selection_start) / 1e9,
            "refit_seconds": (refit_end - refit_start) / 1e9,
            "selection_scope": (
                "Inner scaler/features, fresh model/optimizer, all selection epochs "
                "and tail scoring, including existing checks and progress callbacks"
            ),
            "refit_scope": (
                "Full-prefix scaler/features, same-seed fresh model/optimizer and "
                "selected-epoch training, including existing checks and progress callbacks"
            ),
        }
    return LSTMFit(refit_model, full_scaler, metadata, selection_model=model)


def fit_fixed_epochs(values, observed, timestamps, fit_end, *, hidden_units,
                     dropout, seed, settings, epochs, progress=None):
    """Fit a fixed, previously selected epoch count without any stopping tail."""
    budget = _settings(settings, hidden_units, dropout, seed)
    epochs = _positive_integer(epochs, "epochs", budget["maximum_epochs"])
    past, flags, clock = _prefix(values, observed, timestamps, fit_end)
    scaler = fit_scaler(past, flags, fit_end)
    train = make_examples(past, flags, clock, 0, fit_end)
    arrays = _training_arrays(train, scaler)
    model, optimizer = _new_model(hidden_units, dropout, seed, settings)
    initial = state_fingerprint(model)
    history = _fit_epochs(model, optimizer, arrays, epochs, settings["batch_size"], progress)
    return LSTMFit(model, scaler, {
        "mode": "fixed_epochs_without_early_stopping", "hidden_units": int(hidden_units),
        "dropout": float(dropout), "seed": int(seed), "fit_interval": [0, fit_end],
        "fit_count": len(train.targets), "scaler": asdict(scaler), "epochs": epochs,
        "history": history, "initial_state_sha256": initial,
        "optimizer_defaults": dict(optimizer.defaults), "early_stopping": False,
        "component_fit_counts": {"full_prefix_fit": 1},
    })
