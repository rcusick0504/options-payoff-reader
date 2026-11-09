"""Fetch and audit an explicit, source-tracked set of public educational images.

Downloaded images are local evaluation assets, not redistributed training data.
No network calls occur unless the fetch command is explicitly invoked.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

SOURCES = Path("data/web_chart_sources.json")
ROOT = Path("data/web_charts")


def sources(split=None):
    rows = json.loads(SOURCES.read_text())
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source identifiers")
    publishers = {}
    for row in rows:
        if not row["id"].replace("-", "").isalnum():
            raise ValueError("Invalid source identifier")
        prior = publishers.setdefault(row["publisher"], row["split"])
        if prior != row["split"]:
            raise ValueError("Publishers must stay in one split")
    return [r for r in rows if split is None or r["split"] == split]


def fetch(split=None):
    ROOT.mkdir(parents=True, exist_ok=True)
    for row in sources(split):
        path = ROOT / f"{row['id']}.png"
        metadata = ROOT / f"{row['id']}.json"
        if path.exists() and metadata.exists():
            continue
        request = Request(row["image_url"], headers={"User-Agent": "OptionsPayoffReader-EducationalAudit/1.0"})
        with urlopen(request, timeout=30) as response:
            data = response.read(20_000_001)
        if len(data) > 20_000_000:
            raise ValueError("Source image exceeds 20 MB")
        with Image.open(BytesIO(data)) as image:
            image.convert("RGB").save(path)
        metadata.write_text(json.dumps(dict(source=row, downloaded_at=datetime.now(timezone.utc).isoformat(), source_sha256=hashlib.sha256(data).hexdigest(), image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()), indent=2) + "\n")
        print(f"Downloaded {row['id']}", flush=True)


def audit(split, output, numerical=False):
    from .measure import measure
    from .payoffs import CLASSES
    from .predict import Predictor

    model = Predictor()
    rows = []
    for row in sources(split):
        path = ROOT / f"{row['id']}.png"
        if not path.exists():
            raise FileNotFoundError(f"Run python -m src.webcharts fetch --split {split}")
        with Image.open(path) as image:
            prediction = model.predict(image)
            category = prediction.get("category")
            result = dict(id=row["id"], publisher=row["publisher"], source_url=row["source_url"], image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), expected=CLASSES[row["label"]], prediction=prediction, correct=category == CLASSES[row["label"]])
            if numerical and category in CLASSES:
                result["measurements"] = measure(image, CLASSES.index(category), vertical_axis=row.get("vertical_axis","auto"))
                result["expected_values"] = row.get("expected")
            rows.append(result)
            print(row["id"], category, result["correct"], flush=True)
    payload = dict(split=split, model=model.name, checkpoint_sha256=hashlib.sha256(Path(f"models/{model.name}.pt").read_bytes()).hexdigest(), correct=sum(r["correct"] for r in rows), total=len(rows), rows=rows)
    Path(output).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fetch", "audit"])
    parser.add_argument("--split", choices=["development", "holdout"], default="development")
    parser.add_argument("--numerical", action="store_true")
    parser.add_argument("--output", default="docs/results/web_multisource_audit.json")
    args = parser.parse_args()
    if args.stage == "fetch":
        fetch(args.split)
    else:
        audit(args.split, args.output, args.numerical)
