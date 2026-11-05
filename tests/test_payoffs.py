"""Independent equation and slope checks. Codex-generated; see ATTRIBUTION.md."""

import numpy as np
import pytest
from src.payoffs import Leg, construct, profit


def test_leg_sign_premium():
    np.testing.assert_allclose(
        Leg("call", 100, 5, 1).profit([80, 100, 110]), [-5, -5, 5]
    )
    np.testing.assert_allclose(
        Leg("put", 100, 5, -2).profit([80, 100, 110]), [-30, 10, 10]
    )


@pytest.mark.parametrize(
    "label,expected",
    [
        (0, [-5, -5, -5, 5, 15]),
        (1, [15, 5, -5, -5, -5]),
        (2, [-5, -5, 5, 15, 15]),
        (3, [15, 15, 5, -5, -5]),
        (4, [10, 0, -10, 0, 10]),
        (5, [0, -10, -10, -10, 0]),
        (6, [-5, 5, 15, 5, -5]),
        (7, [-10, 0, 10, 0, -10]),
    ],
)
def test_values_at_strikes_and_tails(label, expected):
    np.testing.assert_allclose(
        profit(np.array([80, 90, 100, 110, 120]), construct(label)), expected
    )


@pytest.mark.parametrize(
    "label,slopes",
    [
        (0, [0, 0, 1, 1]),
        (1, [-1, -1, 0, 0]),
        (2, [0, 1, 1, 0]),
        (3, [0, -1, -1, 0]),
        (4, [-1, -1, 1, 1]),
        (5, [-1, 0, 0, 1]),
        (6, [0, 1, -1, 0]),
        (7, [1, 1, -1, -1]),
    ],
)
def test_piecewise_slopes(label, slopes):
    x = np.array([70.0, 95.0, 105.0, 130.0])
    np.testing.assert_allclose(
        (profit(x + 0.001, construct(label)) - profit(x, construct(label))) / 0.001,
        slopes,
        atol=1e-8,
    )


def test_continuity_and_invalid_parameters():
    for label in range(8):
        legs = construct(label)
        for leg in legs:
            assert (
                abs(
                    profit(np.array([leg.strike - 1e-7]), legs)[0]
                    - profit(np.array([leg.strike + 1e-7]), legs)[0]
                )
                < 1e-6
            )
    with pytest.raises(ValueError):
        construct(0, width=20, premium=20)
