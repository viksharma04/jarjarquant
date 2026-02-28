import numpy as np
import pytest
from jarjarquant.transforms import (
    root_transform,
    log_transform,
    sigmoid_transform,
    exp_smoothing,
    apply_transform,
)


def test_root_transform_preserves_sign():
    data = np.array([-4.0, 0.0, 9.0])
    result = root_transform(data, degree=2)
    assert result[0] < 0  # negative preserved
    assert result[1] == 0.0
    assert result[2] > 0


def test_log_transform_handles_negatives():
    data = np.array([-1.0, 0.0, 1.0, 10.0])
    result = log_transform(data)
    assert not np.any(np.isnan(result))


def test_sigmoid_transform_bounds():
    data = np.array([-100.0, 0.0, 100.0])
    result = sigmoid_transform(data)
    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)


def test_exp_smoothing():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = exp_smoothing(data, span=3)
    assert len(result) == len(data)
    assert result[-1] != data[-1]  # smoothed, not identity


def test_apply_transform_dispatches():
    data = np.array([1.0, 4.0, 9.0])
    result = apply_transform(data, "root")
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0]))


def test_apply_transform_tanh_alias():
    """'tanh' should be an alias for 'sigmoid' for backward compat."""
    data = np.array([-100.0, 0.0, 100.0])
    result_sigmoid = apply_transform(data.copy(), "sigmoid")
    result_tanh = apply_transform(data.copy(), "tanh")
    np.testing.assert_array_equal(result_sigmoid, result_tanh)


def test_apply_transform_unknown_raises():
    with pytest.raises(ValueError, match="Unknown transform"):
        apply_transform(np.array([1.0]), "nonexistent")
