"""Single-layer CPU LSTM; shared code owns scaling and final traffic bounds."""

from numbers import Integral, Real

import numpy as np
import torch
from torch import nn


def _integer(value, name, minimum=1):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def configure_cpu(seed):
    """Set repeatable initialization and the requested PyTorch CPU controls.

    Startup environment controls are separately verified by the command runner.
    Inter-operation threads can only be set before parallel work starts; an
    incompatible already-running process must fail instead of hiding that fact.
    """
    seed = _integer(seed, "seed", 0)
    if seed > 2**32 - 1:
        raise ValueError("seed must fit an unsigned 32-bit integer")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    if (torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1
            or not torch.are_deterministic_algorithms_enabled()):
        raise RuntimeError("PyTorch CPU determinism controls were not applied")


class TrafficLSTM(nn.Module):
    """Read 168 traffic/missing-flag pairs and four target-clock features."""

    def __init__(self, hidden_units, output_dropout):
        super().__init__()
        self.hidden_units = _integer(hidden_units, "hidden_units")
        if (isinstance(output_dropout, (bool, np.bool_))
                or not isinstance(output_dropout, Real)
                or not np.isfinite(output_dropout)
                or not 0 <= output_dropout < 1):
            raise ValueError("output_dropout must be finite and in [0, 1)")
        self.output_dropout = float(output_dropout)
        self.lstm = nn.LSTM(
            2, self.hidden_units, num_layers=1, bias=True, batch_first=True,
            dropout=0.0, bidirectional=False, proj_size=0,
            device="cpu", dtype=torch.float32,
        )
        # Output dropout acts on the final hidden vector, not on calendar values.
        self.dropout = nn.Dropout(self.output_dropout)
        self.head = nn.Linear(self.hidden_units + 4, 1, bias=True,
                              device="cpu", dtype=torch.float32)

    def forward(self, sequence, calendar):
        if (not isinstance(sequence, torch.Tensor)
                or not isinstance(calendar, torch.Tensor)
                or sequence.ndim != 3 or tuple(sequence.shape[1:]) != (168, 2)
                or tuple(calendar.shape) != (len(sequence), 4)):
            raise ValueError("Expected sequence (n,168,2) and calendar (n,4)")
        if (sequence.device.type != "cpu" or calendar.device.type != "cpu"
                or sequence.dtype != torch.float32
                or calendar.dtype != torch.float32):
            raise ValueError("LSTM inputs must be float32 CPU tensors")
        # Omitting h0/c0 creates new zero states for every independent window.
        _, (hidden, _) = self.lstm(sequence)
        combined = torch.cat((self.dropout(hidden[-1]), calendar), dim=1)
        return self.head(combined).squeeze(-1)


def build_model(hidden_units, dropout, seed):
    """Initialize one model; production-grid checks belong to the fit pipeline."""
    configure_cpu(seed)
    return TrafficLSTM(hidden_units, dropout)


def _float32(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all() or np.any(np.abs(array) > np.finfo(np.float32).max):
        raise ValueError(f"{name} must be representable as finite float32 values")
    # A private contiguous copy also accepts reversed/read-only input arrays.
    return np.array(array, dtype=np.float32, order="C", copy=True)


def _inputs(sequence, calendar, allow_empty):
    sequence = _float32(sequence, "Sequence")
    calendar = _float32(calendar, "Calendar")
    if sequence.ndim != 3 or sequence.shape[1:] != (168, 2):
        raise ValueError("Sequence must have shape (n,168,2)")
    if calendar.shape != (len(sequence), 4):
        raise ValueError("Calendar must have shape (n,4) matching the sequence")
    if not allow_empty and len(sequence) == 0:
        raise ValueError("Training requires at least one example")
    return torch.from_numpy(sequence), torch.from_numpy(calendar)


def _check_model(model):
    if not isinstance(model, TrafficLSTM):
        raise ValueError("Model must be a TrafficLSTM")
    for parameter in model.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError("LSTM parameters must remain float32 on CPU")
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError("LSTM parameters must be finite")


def train_epoch(model, optimizer, sequence, calendar, targets, batch_size):
    """One chronological pass, including the last partial batch; return MSE.

    Each batch uses fresh hidden states. The returned loss is the sample-weighted
    mean of pre-update batch losses, not a separate post-epoch evaluation.
    """
    _check_model(model)
    batch_size = _integer(batch_size, "batch_size")
    sequence, calendar = _inputs(sequence, calendar, allow_empty=False)
    targets = _float32(targets, "Targets")
    if targets.shape != (len(sequence),):
        raise ValueError("Targets must be one-dimensional and match input rows")
    targets = torch.from_numpy(targets)
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError("optimizer must be a PyTorch optimizer")
    parameters = list(model.parameters())
    optimizer_parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if (len(optimizer_parameters) != len(parameters)
            or {id(p) for p in optimizer_parameters} != {id(p) for p in parameters}):
        raise ValueError("Optimizer must own exactly this model's parameters")
    model.train()
    total_loss = 0.0
    for start in range(0, len(sequence), batch_size):
        stop = min(start + batch_size, len(sequence))
        optimizer.zero_grad(set_to_none=True)
        predicted = model(sequence[start:stop], calendar[start:stop])
        loss = nn.functional.mse_loss(predicted, targets[start:stop], reduction="mean")
        if not bool(torch.isfinite(loss)):
            raise ValueError("Training loss must be finite")
        loss.backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise ValueError("Training gradients must exist and be finite")
        optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in parameters):
            raise ValueError("Updated model parameters must be finite")
        total_loss += float(loss.detach()) * (stop - start)
    return total_loss / len(sequence)


def predict_lstm(model, sequence, calendar, batch_size=32):
    """Return raw standardized predictions; do not invert or clip them here."""
    _check_model(model)
    batch_size = _integer(batch_size, "batch_size")
    sequence, calendar = _inputs(sequence, calendar, allow_empty=True)
    model.eval()
    if len(sequence) == 0:
        return np.empty(0, dtype=np.float64)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(sequence), batch_size):
            predicted = model(sequence[start:start + batch_size],
                              calendar[start:start + batch_size])
            predictions.append(predicted.numpy().astype(np.float64))
    result = np.concatenate(predictions)
    if result.shape != (len(sequence),) or not np.isfinite(result).all():
        raise ValueError("LSTM predictions must be finite and one-dimensional")
    return result
