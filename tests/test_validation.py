import math

import pytest

from ratchet.validation import finite_number, mapping, nonblank, rate, whole_number


@pytest.mark.parametrize("value", [True, False, None, "bad", math.nan, math.inf, -math.inf])
def test_finite_number_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="field must be a finite number"):
        finite_number(value, "field")


def test_numeric_helpers_normalize_and_bound_values():
    assert finite_number("1.5", "field") == 1.5
    assert rate("0.25", "rate") == 0.25
    assert whole_number("6", "rounds", minimum=1) == 6
    for value in (-0.1, 1.1):
        with pytest.raises(ValueError, match="between 0 and 1"):
            rate(value, "rate")
    for value in (True, 0, 1.5):
        with pytest.raises(ValueError, match="integer"):
            whole_number(value, "rounds", minimum=1)


def test_shape_helpers_are_strict():
    assert mapping({"a": 1}, "config") == {"a": 1}
    assert nonblank(" x ", "name") == "x"
    with pytest.raises(ValueError, match="mapping"):
        mapping([], "config")
    with pytest.raises(ValueError, match="nonblank"):
        nonblank("  ", "name")
