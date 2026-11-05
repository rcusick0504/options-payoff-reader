"""Leakage, geometry, preprocessing and input checks. Codex-generated."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from src.generate import ROOT, BOX, render, style
from src.payoffs import sample_trade
from src.learning import preprocess


def test_split_integrity():
    rows = [json.loads(s) for s in (ROOT / "manifest.jsonl").read_text().splitlines()]
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["sha256"] for r in rows}) == len(rows)
    assert len({json.dumps(r["trade"], sort_keys=True) for r in rows}) == len(rows)
    for split, n in [("train", 1500), ("val", 200), ("test", 200)]:
        rs = [r for r in rows if r["split"] == split]
        assert all(sum(r["label"] == i for r in rs) == n for i in range(8))
    for r in rows:
        assert all(
            r["trade"]["xlim"][0] < l["strike"] < r["trade"]["xlim"][1]
            for l in r["trade"]["legs"]
        )


def test_preprocess_crop_matches_contract():
    im = Image.new("RGB", (200, 200), "red")
    x = preprocess(im, crop=True, gray=True)
    assert tuple(x.shape) == (3, 64, 64)
    assert np.array_equal(x[0], x[1])
    assert 0 <= x.min() <= x.max() <= 1


def test_render_reproducible():
    trade = sample_trade(0, np.random.default_rng(1))
    s = style(np.random.default_rng(2))
    assert render(trade, s).tobytes() == render(trade, s).tobytes()
