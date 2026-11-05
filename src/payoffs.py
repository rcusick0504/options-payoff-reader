"""Analytic educational profit equations. Generated with Codex; see ATTRIBUTION.md.

Units are per underlying share, ignoring fees, financing, and contract multipliers.
Premiums are synthetic assumptions, never live quotes.
"""

from dataclasses import dataclass, asdict
import numpy as np

CLASSES = [
    "Long call",
    "Long put",
    "Bull spread",
    "Bear spread",
    "Long straddle",
    "Long strangle",
    "Long butterfly",
    "Short straddle",
]
EXPLANATIONS = [
    "Flat loss below the strike, then increasing profit as expiration price rises.",
    "Profit rises as expiration price falls below the strike; loss is flat above it.",
    "An increasing segment between two flat tails; both gain and loss are bounded.",
    "A decreasing segment between two flat tails; both gain and loss are bounded.",
    "A V-shaped profile with its minimum at a common strike.",
    "Two rising outer wings with a flat loss region between two strikes.",
    "A central peak between flat loss tails, constructed with equally spaced calls.",
    "An inverted V with a central maximum and losses toward both ends of the chart.",
]


@dataclass(frozen=True)
class Leg:
    kind: str
    strike: float
    premium: float
    quantity: int

    def __post_init__(self):
        if self.kind not in ("call", "put") or self.strike <= 0 or self.premium < 0:
            raise ValueError("Invalid option leg")

    def profit(self, prices):
        prices = np.asarray(prices, dtype=float)
        intrinsic = np.maximum(
            prices - self.strike if self.kind == "call" else self.strike - prices, 0
        )
        return self.quantity * (intrinsic - self.premium)


def profit(prices, legs):
    return sum((leg.profit(prices) for leg in legs), np.zeros_like(prices, dtype=float))


def construct(label, center=100.0, width=20.0, premium=5.0):
    """Positive, nondegenerate educational constructions with explicit leg premiums.

    Spread debit = premium (strictly less than width); butterfly debit = premium.
    Strangle strikes center +/- width/2. Butterfly strikes center +/- width.
    """
    if label not in range(8) or not 0 < premium < width or center <= width:
        raise ValueError("Invalid strategy parameters")
    c, w, p = center, width, premium
    recipes = [
        [Leg("call", c, p, 1)],
        [Leg("put", c, p, 1)],
        [Leg("call", c - w / 2, 2 * p, 1), Leg("call", c + w / 2, p, -1)],
        [Leg("put", c + w / 2, 2 * p, 1), Leg("put", c - w / 2, p, -1)],
        [Leg("call", c, p, 1), Leg("put", c, p, 1)],
        [Leg("put", c - w / 2, p, 1), Leg("call", c + w / 2, p, 1)],
        [
            Leg("call", c - w, 4 * p, 1),
            Leg("call", c, 2 * p, -2),
            Leg("call", c + w, p, 1),
        ],
        [Leg("call", c, p, -1), Leg("put", c, p, -1)],
    ]
    return recipes[label]


def sample_trade(label, rng):
    center = float(rng.uniform(75, 150))
    width = float(center * rng.uniform(0.12, 0.23))
    premium = float(width * rng.uniform(0.12, 0.32))
    legs = construct(label, center, width, premium)
    # Identical x-range distribution for all classes, all strikes visible.
    left = center - width * float(rng.uniform(2.1, 3.0))
    right = center + width * float(rng.uniform(2.1, 3.0))
    return dict(
        center=center,
        width=width,
        premium=premium,
        legs=[asdict(x) for x in legs],
        xlim=[left, right],
    )
