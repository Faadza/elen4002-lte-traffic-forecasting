"""LSTM gate arithmetic, statelessness, chronology and training contracts."""

import math

import numpy as np
import pytest
import torch

from models.lstm import build_model, configure_cpu, predict_lstm, train_epoch


def arrays(n=3):
    return (np.zeros((n, 168, 2), dtype=np.float32),
            np.zeros((n, 4), dtype=np.float32),
            np.zeros(n, dtype=np.float32))


def zero_parameters(model):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()


def test_declared_architecture_and_cpu_controls():
    model = build_model(16, .2, 2026)
    assert model.hidden_units == 16 and model.output_dropout == .2
    assert model.lstm.input_size == 2 and model.lstm.num_layers == 1
    assert model.lstm.batch_first and not model.lstm.bidirectional
    assert model.lstm.bias and model.lstm.proj_size == 0 and model.lstm.dropout == 0
    assert model.dropout.p == .2
    assert model.head.in_features == 20 and model.head.out_features == 1
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    assert torch.are_deterministic_algorithms_enabled()


def test_one_unit_all_gates_and_final_head_match_scalar_recurrence():
    model = build_model(1, 0, 2026)
    zero_parameters(model)
    # Gate order is input, forget, candidate, output; both bias vectors count.
    weights = [.2, -.3, .4, .1]
    recurrent = [.1, .2, -.1, .3]
    first_bias = [.05, -.1, .2, -.2]
    second_bias = [-.02, .04, -.03, .06]
    with torch.no_grad():
        model.lstm.weight_ih_l0[:, 0] = torch.tensor(weights)
        model.lstm.weight_ih_l0[:, 1] = torch.tensor([.1, -.2, .3, -.1])
        model.lstm.weight_hh_l0[:, 0] = torch.tensor(recurrent)
        model.lstm.bias_ih_l0[:] = torch.tensor(first_bias)
        model.lstm.bias_hh_l0[:] = torch.tensor(second_bias)
        model.head.weight[:] = torch.tensor([[1.7, .2, -.3, .4, -.5]])
        model.head.bias[:] = -.7
    sequence, calendar, _ = arrays(1)
    sequence[0, :, 0] = np.linspace(-.4, .8, 168, dtype=np.float32)
    sequence[0, ::7, 1] = 1
    calendar[:] = [.1, -.2, .3, -.4]
    hidden = cell = 0.0
    for traffic, missing in sequence[0]:
        gate = [weights[j] * float(traffic) + [.1, -.2, .3, -.1][j] * float(missing)
                + recurrent[j] * hidden + first_bias[j] + second_bias[j] for j in range(4)]
        sigmoid = lambda value: 1 / (1 + math.exp(-value))
        cell = sigmoid(gate[1]) * cell + sigmoid(gate[0]) * math.tanh(gate[2])
        hidden = sigmoid(gate[3]) * math.tanh(cell)
    expected = 1.7 * hidden + sum(a * float(b) for a, b in zip([.2, -.3, .4, -.5], calendar[0])) - .7
    assert predict_lstm(model, sequence, calendar)[0] == pytest.approx(expected, abs=2e-7)


def test_output_dropout_leaves_calendar_values_unchanged_and_raw_negative_allowed():
    model = build_model(2, .2, 2026)
    zero_parameters(model)
    with torch.no_grad():
        model.head.weight[0, 2:] = torch.tensor([1., 2., 3., 4.])
        model.head.bias[:] = -10
    sequence, calendar, _ = arrays()
    calendar[:, :] = [.1, .2, .3, .4]
    model.train()
    for _ in range(2):
        predicted = model(torch.from_numpy(sequence), torch.from_numpy(calendar))
        torch.testing.assert_close(predicted, torch.full((3,), -7.))
    result = predict_lstm(model, sequence, calendar)
    assert result.dtype == np.float64 and np.all(result < 0)
    np.testing.assert_array_equal(result, [-7., -7., -7.])


def test_windows_have_no_carried_state_and_inference_is_repeatable():
    model = build_model(3, .2, 2026)
    sequence, calendar, _ = arrays(4)
    sequence[:, :, 0] = np.arange(4)[:, None] * .2
    original = sequence.copy()
    before = {name: p.clone() for name, p in model.state_dict().items()}
    together = predict_lstm(model, sequence, calendar, 2)
    separately = np.concatenate([predict_lstm(model, sequence[i:i+1], calendar[i:i+1]) for i in range(4)])
    np.testing.assert_allclose(together, separately, atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(together, predict_lstm(model, sequence, calendar, 2))
    np.testing.assert_array_equal(sequence, original)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert not model.training


def test_chronological_batches_include_remainder_and_weight_loss_by_examples():
    model = build_model(1, 0, 2026)
    zero_parameters(model)
    sequence, calendar, targets = arrays(5)
    calendar[:, 0] = np.arange(5)
    targets[:] = np.arange(1, 6)
    seen = []
    handle = model.register_forward_pre_hook(lambda _, inputs: seen.append(inputs[1][:, 0].tolist()))
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    loss = train_epoch(model, optimizer, sequence, calendar, targets, 2)
    handle.remove()
    assert seen == [[0., 1.], [2., 3.], [4.]]
    assert loss == pytest.approx(11.)  # (1+4+9+16+25)/5, not mean of batch means.
    assert model.training
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_adam_learns_toy_target_and_repeats_with_same_seed_and_dropout():
    sequence, calendar, targets = arrays(9)
    sequence[:, :, 0] = np.linspace(-.2, .2, 9)[:, None]
    calendar[:, 0] = np.linspace(-1, 1, 9)
    targets[:] = .7 + .3 * calendar[:, 0]

    def fit():
        model = build_model(3, .2, 2026)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        before = np.mean((predict_lstm(model, sequence, calendar) - targets)**2)
        losses = [train_epoch(model, optimizer, sequence, calendar, targets, 4) for _ in range(15)]
        predicted = predict_lstm(model, sequence, calendar)
        return model, before, losses, predicted

    first, before, losses, predicted = fit()
    second, _, repeated_losses, repeated_prediction = fit()
    assert np.mean((predicted - targets)**2) < before
    np.testing.assert_array_equal(losses, repeated_losses)
    np.testing.assert_array_equal(predicted, repeated_prediction)
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("hidden,dropout,seed", [(0, 0, 1), (True, 0, 1), (1.5, 0, 1),
    (1, -.1, 1), (1, 1, 1), (1, np.nan, 1), (1, True, 1),
    (1, 0, -1), (1, 0, True), (1, 0, 1.5), (1, 0, 2**32)])
def test_invalid_architecture_or_seed_rejected(hidden, dropout, seed):
    with pytest.raises(ValueError):
        build_model(hidden, dropout, seed)


@pytest.mark.parametrize("bad", [np.zeros((2, 167, 2)), np.zeros((2, 168)),
    np.zeros((2, 168, 3)), np.full((2, 168, 2), np.nan),
    np.full((2, 168, 2), 1e300), np.full((2, 168, 2), 1j),
    np.full((2, 168, 2), "1")])
def test_invalid_sequences_rejected(bad):
    model = build_model(1, 0, 2026)
    with pytest.raises(ValueError):
        predict_lstm(model, bad, np.zeros((2, 4)))


@pytest.mark.parametrize("bad", [np.zeros((1, 4)), np.zeros((2, 5)),
    np.zeros(4), np.full((2, 4), np.inf)])
def test_invalid_calendar_rejected(bad):
    model = build_model(1, 0, 2026)
    with pytest.raises(ValueError):
        predict_lstm(model, arrays(2)[0], bad)


@pytest.mark.parametrize("bad", [np.zeros((2, 1)), np.zeros(1), [0, np.nan], [1e300, 0]])
def test_targets_cannot_broadcast_or_overflow(bad):
    model = build_model(1, 0, 2026)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    sequence, calendar, _ = arrays(2)
    with pytest.raises(ValueError):
        train_epoch(model, optimizer, sequence, calendar, bad, 2)


def test_empty_prediction_but_not_training():
    model = build_model(1, 0, 2026)
    sequence, calendar, targets = arrays(0)
    assert predict_lstm(model, sequence, calendar).shape == (0,)
    with pytest.raises(ValueError, match="at least one"):
        train_epoch(model, torch.optim.Adam(model.parameters()), sequence, calendar, targets, 2)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_sizes_rejected(batch_size):
    model = build_model(1, 0, 2026)
    sequence, calendar, _ = arrays(2)
    with pytest.raises(ValueError):
        predict_lstm(model, sequence, calendar, batch_size)


def test_non_float32_or_nonfinite_model_and_wrong_optimizer_rejected():
    model = build_model(1, 0, 2026)
    other = build_model(1, 0, 2026)
    sequence, calendar, targets = arrays(2)
    with pytest.raises(ValueError, match="exactly this model"):
        train_epoch(model, torch.optim.Adam(other.parameters()), sequence, calendar, targets, 2)
    model.double()
    with pytest.raises(ValueError, match="float32"):
        predict_lstm(model, sequence, calendar)
    model.float()
    with torch.no_grad():
        model.head.bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        predict_lstm(model, sequence, calendar)


def test_reversed_readonly_inputs_are_accepted_without_mutation():
    model = build_model(1, 0, 2026)
    sequence, calendar, _ = arrays(2)
    sequence.flags.writeable = False
    calendar.flags.writeable = False
    np.testing.assert_array_equal(predict_lstm(model, sequence[::-1], calendar[::-1]),
                                  predict_lstm(model, sequence.copy(), calendar.copy()))
