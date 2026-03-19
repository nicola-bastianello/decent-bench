import numpy as np
import pytest

from decent_bench.costs import BoxIndicator, NonnegativeIndicator
from decent_bench.costs._indicator import IndicatorCost
from decent_bench.utils.array import Array


def test_box_constructor_rejects_inverted_bounds() -> None:
    lower = Array(np.array([0.0, 3.0]))
    upper = Array(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="must be less than or equal"):
        BoxIndicator(lower, upper)


def test_box_belongs_to_set_returns_python_bool() -> None:
    lower = Array(np.array([0.0, -1.0]))
    upper = Array(np.array([1.0, 2.0]))
    indicator = BoxIndicator(lower, upper)

    inside = Array(np.array([0.2, 1.5]))
    outside = Array(np.array([-0.1, 1.5]))

    inside_result = indicator.belongs_to_set(inside)
    outside_result = indicator.belongs_to_set(outside)

    assert isinstance(inside_result, bool)
    assert isinstance(outside_result, bool)
    assert inside_result is True
    assert outside_result is False


def test_nonnegative_belongs_to_set_returns_python_bool() -> None:
    indicator = NonnegativeIndicator((3,))

    inside = Array(np.array([0.0, 1.0, 2.0]))
    outside = Array(np.array([0.0, -1.0, 2.0]))

    inside_result = indicator.belongs_to_set(inside)
    outside_result = indicator.belongs_to_set(outside)

    assert isinstance(inside_result, bool)
    assert isinstance(outside_result, bool)
    assert inside_result is True
    assert outside_result is False


def test_indicator_base_behavior() -> None:
    indicator: IndicatorCost = NonnegativeIndicator((2,))
    x = Array(np.array([1.0, -1.0]))

    assert np.isnan(indicator.m_smooth)
    assert indicator.m_cvx == 0
    assert indicator.function(Array(np.array([1.0, 0.0]))) == 0
    assert indicator.function(x) == np.inf


def test_indicator_proximal_equals_projection() -> None:
    indicator = NonnegativeIndicator((3,))
    x = Array(np.array([-1.0, 0.5, -0.2]))

    projection = indicator.projection(x)
    proximal = indicator.proximal(x, 0.1)

    np.testing.assert_allclose(np.asarray(projection), np.asarray(proximal))
