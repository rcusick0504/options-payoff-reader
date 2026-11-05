"""Deterministic chart generator. AI-generated with Codex; see ATTRIBUTION.md."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .payoffs import CLASSES, Leg, sample_trade, profit

ROOT = Path("data/generated")
BOX = (
    0.14,
    0.09,
    0.94,
    0.83,
)  # normalized PIL left/top/right/bottom; shared crop contract


def style(rng, ood=False):
    return dict(
        width=int(rng.choice([192, 224, 256, 288])),
        height=int(rng.choice([176, 208, 240])),
        background=str(
            rng.choice(
                ["#ffffff", "#f4f4f0", "#edf3f7"] if not ood else ["#202734", "#fff1d2"]
            )
        ),
        line=str(
            rng.choice(
                ["#175da8", "#a32937", "#207645", "#642f92"]
                if not ood
                else ["#e67e22", "#dc55a2", "#26aaae"]
            )
        ),
        thickness=int(rng.choice([2, 3, 4] if not ood else [1, 5])),
        grid=bool(rng.integers(2)),
        font_size=int(rng.choice([9, 11, 13])),
        font=str(rng.choice(["regular", "bold"])),
        ood=ood,
    )


def geometry(trade):
    legs = [Leg(**x) for x in trade["legs"]]
    x = np.sort(
        np.unique(np.r_[np.linspace(*trade["xlim"], 161), [l.strike for l in legs]])
    )
    y = profit(x, legs)
    span = max(float(np.ptp(y)), 1.0)
    return (
        x,
        y,
        [min(float(y.min()), 0) - 0.12 * span, max(float(y.max()), 0) + 0.12 * span],
    )


def render(trade, s):
    w, h = s["width"], s["height"]
    im = Image.new("RGB", (w, h), s["background"])
    d = ImageDraw.Draw(im)
    x, y, ylim = geometry(trade)
    l, t, r, b = [BOX[0] * w, BOX[1] * h, BOX[2] * w, BOX[3] * h]
    xx = l + (x - x.min()) / np.ptp(x) * (r - l)
    yy = b - (y - ylim[0]) / (ylim[1] - ylim[0]) * (b - t)
    dark = s["background"] == "#202734"
    ink = "#dddddd" if dark else "#555555"
    font = ImageFont.load_default(size=s["font_size"])
    # Font variation is independent of class. Built-in bitmap font is portable.
    if s["font"] == "bold":
        font = ImageFont.load_default(size=s["font_size"] + 1)
    for v in np.linspace(*trade["xlim"], 5):
        p = l + (v - x.min()) / np.ptp(x) * (r - l)
        if s["grid"]:
            d.line((p, t, p, b), fill="#657080" if dark else "#d7dce0")
        text = f"{v:.0f}"
        tw = d.textbbox((0, 0), text, font=font)[2]
        d.text((min(w - tw - 1, max(0, p - tw / 2)), b + 3), text, fill=ink, font=font)
    for v in np.linspace(*ylim, 5):
        p = b - (v - ylim[0]) / (ylim[1] - ylim[0]) * (b - t)
        if s["grid"]:
            d.line((l, p, r, p), fill="#657080" if dark else "#d7dce0")
        d.text((1, p - 4), f"{v:.0f}", fill=ink, font=font)
    d.line((l, t, l, b, r, b), fill=ink, width=1)
    zero = b - (0 - ylim[0]) / (ylim[1] - ylim[0]) * (b - t)
    d.line((l, zero, r, zero), fill=ink, width=1)
    d.line(list(zip(xx, yy)), fill=s["line"], width=s["thickness"])
    d.text((w * 0.32, h * 0.92), "Expiration price", fill=ink, font=font)
    d.text((2, 1), "P/L", fill=ink, font=font)
    return im


def independent_render(trade, s):
    """Second rendering implementation, Matplotlib; shared analytic trade equations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO

    x, y, ylim = geometry(trade)
    fig = plt.figure(
        figsize=(s["width"] / 100, s["height"] / 100),
        dpi=100,
        facecolor=s["background"],
    )
    ax = fig.add_axes(
        [BOX[0], 1 - BOX[3], BOX[2] - BOX[0], BOX[3] - BOX[1]],
        facecolor=s["background"],
    )
    ax.plot(x, y, color=s["line"], linewidth=s["thickness"] / 2)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.set(xlim=trade["xlim"], ylim=ylim, xlabel="Expiration price", ylabel="P/L")
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.grid(True, alpha=0.3) if s["grid"] else ax.grid(False)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def preview():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(386)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for label, ax in enumerate(axes.flat):
        ax.imshow(render(sample_trade(label, rng), style(rng)))
        ax.set_title(CLASSES[label])
        ax.axis("off")
    fig.suptitle("Eight educational profit/loss shapes — labels outside model inputs")
    fig.tight_layout()
    fig.savefig("docs/figures/class_preview.png", dpi=140)
    plt.close(fig)


def generate():
    if (ROOT / "manifest.jsonl").exists():
        raise RuntimeError(
            "Dataset exists; preserve it or explicitly remove before regeneration."
        )
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    hashes = set()
    for split, n, offset in [
        ("train", 1500, 0),
        ("val", 200, 1),
        ("test", 200, 2),
        ("ood", 100, 3),
        ("independent", 10, 4),
        ("ambiguous", 5, 5),
    ]:
        (ROOT / split).mkdir(exist_ok=True)
        for label in range(8):
            for i in range(n):
                seed = 386 + offset * 1000000 + label * 10000 + i
                # Separate streams: trade sampling cannot drive style-label correlation.
                rng = np.random.default_rng(seed)
                srng = np.random.default_rng(seed + 90000000)
                trade = sample_trade(label, rng)
                s = style(srng, split == "ood")
                im = (
                    independent_render(trade, s)
                    if split == "independent"
                    else render(trade, s)
                )
                if split == "ambiguous":
                    # Remove most geometry; these carry no scoreable class label.
                    im = im.crop((0, 0, int(im.width * 0.25), im.height)).resize(
                        im.size
                    )
                ident = f"{split}-{label}-{i:04d}"
                image_hash = hashlib.sha256(im.tobytes()).hexdigest()
                if image_hash in hashes:
                    raise AssertionError("Duplicate rendered image")
                hashes.add(image_hash)
                path = f"{split}/{ident}.png"
                im.save(ROOT / path)
                rows.append(
                    dict(
                        id=ident,
                        trade_id=ident,
                        label=label if split != "ambiguous" else None,
                        source_label=label,
                        split=split,
                        seed=seed,
                        trade=trade,
                        style=s,
                        path=path,
                        sha256=image_hash,
                    )
                )
        print(f"{split}: {n*8} images", flush=True)
    with (ROOT / "manifest.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {
        s: sum(r["split"] == s for r in rows)
        for s in ["train", "val", "test", "ood", "independent", "ambiguous"]
    }
    summary["manifest_sha256"] = hashlib.sha256(
        (ROOT / "manifest.jsonl").read_bytes()
    ).hexdigest()
    Path("docs/dataset_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preview", action="store_true")
    a = p.parse_args()
    preview() if a.preview else generate()
