"""Versioned multi-format training images. Generated with Codex; see ATTRIBUTION.md.

50 training families, plus 10 held-out layout/theme combinations. Simulated pen
strokes are not a substitute for photographs of real handwriting.
"""
from io import BytesIO
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from .generate import geometry
from .payoffs import sample_trade

THEMES = [
    ("textbook", "#ffffff", "#222222", "#333333"),
    ("terminal", "#151c27", "#4ee3bf", "#d4dce8"),
    ("spreadsheet", "#ffffff", "#2672ba", "#555555"),
    ("cream-paper", "#fff5dd", "#914124", "#564e43"),
    ("blueprint", "#153c70", "#ffffff", "#c7ddfa"),
    ("graph-paper", "#f5fbf6", "#793ca7", "#324d42"),
    ("pen", "#faf8f0", "#1d3c78", "#474747"),
    ("pencil", "#eeeeea", "#343434", "#737373"),
    ("chalkboard", "#203f36", "#f4f0cc", "#c6d3c7"),
    ("presentation", "#f2eff9", "#b72a69", "#42374f"),
]


def family_name(family):
    return f"{THEMES[family % 10][0]}-layout-{family // 10 + 1}"


def font(size, variant):
    # Matplotlib's distributed fonts make this portable across operating systems.
    import matplotlib
    names = ["DejaVuSans.ttf", "DejaVuSerif.ttf", "DejaVuSansMono.ttf"]
    path = Path(matplotlib.get_data_path()) / "fonts/ttf" / names[variant % 3]
    return ImageFont.truetype(str(path), size)


def render_diverse(trade, family, seed, size=None):
    if not 0 <= family < 60:
        raise ValueError("Family must be between 0 and 59")
    rng = np.random.default_rng(seed)
    theme, bg, curve, ink = THEMES[family % 10]
    layout = family // 10
    w, h = size or (int(rng.integers(360, 741)), int(rng.integers(280, 561)))
    im = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(im)
    f = font(max(11, int(min(w, h) * rng.uniform(.031, .043))), layout)
    # Variable plot location, whitespace and aspect ratio; never a fixed crop.
    l = w * rng.uniform(.13, .22)
    r = w * rng.uniform(.81, .95)
    t = h * rng.uniform(.10, .23)
    b = h * rng.uniform(.73, .85)
    x, y, ylim = geometry(trade)
    xp = l + (x - x.min()) / np.ptp(x) * (r - l)
    yp = b - (y - ylim[0]) / np.ptp(ylim) * (b - t)
    zero = b + ylim[0] / np.ptp(ylim) * (b - t)
    grid = theme in ("spreadsheet", "graph-paper", "blueprint") or layout == 2
    color = tuple(int(v) for v in (.8 * np.array(Image.new("RGB", (1, 1), bg).getpixel((0, 0))) + .2 * np.array(Image.new("RGB", (1, 1), ink).getpixel((0, 0)))))
    if theme == "graph-paper":
        for px in np.arange(l, r, 12):
            d.line((px, t, px, b), fill=color)
        for py in np.arange(t, b, 12):
            d.line((l, py, r, py), fill=color)
    # Decimal values retained in metadata; printed ticks deliberately rounded.
    xticks = np.linspace(*trade["xlim"], int(rng.integers(4, 7)))
    yticks = np.linspace(*ylim, 5)
    for value in xticks:
        px = l + (value - x.min()) / np.ptp(x) * (r - l)
        if grid:
            d.line((px, t, px, b), fill=color)
        d.text((px, b + 8), f"{value:.0f}", font=f, fill=ink, anchor="mt")
        d.line((px, b, px, b + 4), fill=ink)
    for value in yticks:
        py = b - (value - ylim[0]) / np.ptp(ylim) * (b - t)
        if grid:
            d.line((l, py, r, py), fill=color)
        d.text((l - 9, py), f"{value:.0f}", font=f, fill=ink, anchor="rm")
    if layout == 1:
        d.rectangle((l, t, r, b), outline=ink, width=1)
    else:
        d.line((l, t, l, b, r, b), fill=ink, width=1)
    d.line((l, zero, r, zero), fill=ink, width=1)
    thickness = int(rng.integers(2, 6))
    sketch = theme in ("pen", "pencil", "chalkboard")
    if sketch:
        # Smooth low-amplitude jitter preserves actual kinks and flat regions.
        jitter = np.interp(np.linspace(0, 1, len(yp)), np.linspace(0, 1, 24), rng.normal(0, .9, 24))
        yp = yp + jitter
    points = list(zip(xp, yp))
    if layout == 3:
        for j in range(0, len(points) - 1, 6):
            d.line(points[j:j + 5], fill=curve, width=thickness)
    else:
        d.line(points, fill=curve, width=thickness)
    if layout == 4:
        for px, py in points[::20]:
            d.ellipse((px - 3, py - 3, px + 3, py + 3), fill=curve)
    d.text((l, t - f.size * 1.7), str(rng.choice(["Profit / Loss", "P&L at expiration", "Payoff diagram"])), font=f, fill=ink)
    d.text(((l + r) / 2, h - f.size * 1.6), str(rng.choice(["Underlying price", "Price at expiration", "Stock price"])), font=f, fill=ink, anchor="mt")
    if layout == 2:
        d.rectangle((0, 0, w, max(14, t * .35)), fill=color)
    if layout == 5:
        # Reserved layout combination: screenshot border and asymmetric whitespace.
        d.rectangle((3, 3, w - 4, h - 4), outline=ink, width=2)
    if sketch:
        a = np.asarray(im).astype(float)
        a += rng.normal(0, 1.5, (h, w, 1))
        im = Image.fromarray(np.uint8(np.clip(a, 0, 255)))
    if rng.random() < .25:
        im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(.2, .65))))
    if rng.random() < .35:
        buf = BytesIO()
        im.save(buf, "JPEG", quality=int(rng.integers(55, 91)))
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
    return im


def sample(index, split="train"):
    offsets = {"train": 0, "val": 1000000, "test": 2000000, "unseen_val": 3000000, "unseen_test": 4000000}
    seed = 731 + offsets[split] + index
    label = index % 8
    families = range(50) if split in ("train", "val", "test") else range(50, 55) if split == "unseen_val" else range(55, 60)
    family = families[(index // 8) % len(families)]
    trade = sample_trade(label, np.random.default_rng(seed))
    return render_diverse(trade, family, seed + 9000000), label, dict(seed=seed, family=family, family_name=family_name(family), trade=trade)


def preview(path="docs/figures/diverse_formats.jpg"):
    sheet = Image.new("RGB", (5 * 280, 10 * 225), "white")
    d = ImageDraw.Draw(sheet)
    for family in range(50):
        trade = sample_trade(family % 8, np.random.default_rng(500 + family))
        im = render_diverse(trade, family, 900 + family, (560, 390)).resize((280, 195))
        pos = ((family % 5) * 280, (family // 5) * 225)
        sheet.paste(im, pos)
        d.text((pos[0] + 4, pos[1] + 199), family_name(family), fill="black", font=font(12, 0))
    sheet.save(path)
    Path("docs/style_families.json").write_text(json.dumps({"training": [family_name(i) for i in range(50)], "unseen_validation": [family_name(i) for i in range(50, 55)], "unseen_test": [family_name(i) for i in range(55, 60)]}, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    preview()
