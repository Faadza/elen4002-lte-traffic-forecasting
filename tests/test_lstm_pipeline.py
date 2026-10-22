"""Check independent saved-weight equations, stopping audit and evidence safety."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from models.lstm import build_model, predict_lstm
from tools.check_lstm import (GATE_ATOL, GATE_RTOL, check_module, model_weights,
                              predict_saved_weights, restore_weights, verify_stopping_history)


@pytest.mark.parametrize("hidden", [16, 32])
def test_independent_gate_equations_and_json_reload_match_native_lstm(hidden):
    rng = np.random.default_rng(19)
    sequence = rng.normal(size=(3, 168, 2)).astype(np.float32)
    calendar = rng.normal(size=(3, 4)).astype(np.float32)
    model = build_model(hidden, .2, 2026)
    native = predict_lstm(model, sequence, calendar)
    weights = model_weights(model)
    manual = predict_saved_weights(weights, sequence, calendar)
    np.testing.assert_allclose(native, manual, rtol=GATE_RTOL, atol=GATE_ATOL)
    restored = restore_weights(weights, hidden, .2, 2026)
    np.testing.assert_array_equal(predict_lstm(restored, sequence, calendar), native)
    assert sum(p.numel() for p in model.parameters()) == {16: 1301, 32: 4645}[hidden]


def test_hand_gate_recurrence_and_calendar_head_are_reconstructed():
    model = build_model(1, .2, 2026)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # i=f=o=.5 and g=.5, so c168=.5*(1-.5**168), h=.5*tanh(c168).
        model.lstm.bias_ih_l0[2] = np.arctanh(.5)
        model.head.weight[0] = torch.tensor([2., 1., -1., 0., 0.])
        model.head.bias[0] = -1.
    sequence = np.zeros((1, 168, 2), dtype=np.float32)
    calendar = np.array([[.25, .5, 0., 0.]], dtype=np.float32)
    expected = np.tanh(.5 * (1 - .5**168)) + .25 - .5 - 1
    reconstructed = predict_saved_weights(model_weights(model), sequence, calendar)
    np.testing.assert_allclose(reconstructed, [expected], atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(predict_lstm(model, sequence, calendar), [expected], atol=1e-7, rtol=1e-7)
    assert reconstructed[0] < 0  # No model-stage traffic clipping or output activation.


def trace():
    return {"stopping_history": [
        {"epoch": 1, "tail_mae": 3., "best_epoch": 1, "epochs_since_improvement": 0},
        {"epoch": 2, "tail_mae": 2., "best_epoch": 2, "epochs_since_improvement": 0},
        {"epoch": 3, "tail_mae": 2., "best_epoch": 2, "epochs_since_improvement": 1},
        {"epoch": 4, "tail_mae": 4., "best_epoch": 2, "epochs_since_improvement": 2}],
        "selected_epochs": 2, "best_tail_mae": 2., "selection_epochs_run": 4,
        "stop_reason": "patience", "refit_history": [{"epoch": 1}, {"epoch": 2}]}


def test_stopping_audit_accepts_earliest_tie_and_rejects_wrong_refit_count():
    metadata = trace()
    settings = {"maximum_epochs": 10, "patience_epochs": 2}
    verify_stopping_history(metadata, settings)
    altered = deepcopy(metadata)
    altered["selected_epochs"] = 3
    with pytest.raises(AssertionError, match="earliest strict"):
        verify_stopping_history(altered, settings)
    altered = deepcopy(metadata)
    altered["refit_history"].append({"epoch": 3})
    with pytest.raises(AssertionError, match="exactly"):
        verify_stopping_history(altered, settings)


def test_cap_is_a_valid_stop_without_inventing_a_patience_event():
    metadata = trace()
    metadata["stopping_history"] = metadata["stopping_history"][:2]
    metadata["selection_epochs_run"] = 2
    metadata["stop_reason"] = "maximum_epochs"
    verify_stopping_history(metadata, {"maximum_epochs": 2, "patience_epochs": 10})


def test_existing_evidence_is_rejected_before_loading_or_training(tmp_path):
    evidence = tmp_path / "checks.json"
    evidence.write_text("existing evidence")
    with pytest.raises(ValueError, match="new or empty"):
        check_module("data-that-does-not-exist", tmp_path)
    assert evidence.read_text() == "existing evidence"
