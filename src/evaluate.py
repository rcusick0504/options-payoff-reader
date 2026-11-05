"""Final held-out evaluation after selection lock. Codex-generated; see ATTRIBUTION.md."""

import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    classification_report,
    confusion_matrix,
)
from .learning import load_split, records
from .generate import ROOT
from .predict import Predictor
from .payoffs import CLASSES


def evaluate():
    selection = json.loads(Path("models/selection.json").read_text())
    results = {}
    for candidate in selection["candidates"]:
        name = candidate["name"]
        model = Predictor(name)
        c = model.config
        results[name] = {}
        for split in ["test", "ood", "independent"]:
            x, y, rows = load_split(
                split, c.get("crop", False), c.get("gray", False), c.get("size", 64)
            )
            p = np.concatenate([model.probabilities(b) for b in x.split(128)])
            pred = p.argmax(1)
            r = dict(
                n=len(y),
                accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                log_loss=float(log_loss(y, p, labels=list(range(8)))),
                per_class=classification_report(
                    y, pred, target_names=CLASSES, output_dict=True, zero_division=0
                ),
                confusion=confusion_matrix(y, pred, labels=list(range(8))).tolist(),
            )
            results[name][split] = r
            if name == selection["name"]:
                Path(f"docs/results/predictions_{split}.json").write_text(
                    json.dumps(
                        [
                            dict(
                                id=row["id"],
                                path=row["path"],
                                truth=int(y[i]),
                                prediction=int(pred[i]),
                                confidence=float(p[i].max()),
                            )
                            for i, row in enumerate(rows)
                        ],
                        indent=2,
                    )
                )
            print(name, split, r["accuracy"], flush=True)
        im = Image.open(ROOT / records("val")[0]["path"]).convert("RGB")
        for _ in range(10):
            model.predict(im)
        samples = []
        for _ in range(50):
            start = time.perf_counter()
            model.predict(im)
            samples.append((time.perf_counter() - start) * 1000)
        results[name]["latency"] = dict(
            median_ms=float(np.median(samples)),
            p95_ms=float(np.percentile(samples, 95)),
            batch_size=1,
            warmup=10,
            repeats=50,
            method="perf_counter, CPU, preprocessing + prediction; excludes file decoding and model loading",
            hardware=platform.platform(),
            threads=torch.get_num_threads(),
            model_bytes=candidate["model_bytes"],
        )
        Path("docs/results/evaluation.json").write_text(json.dumps(results, indent=2))
    model = Predictor()
    amb = []
    for row in records("ambiguous"):
        result = model.predict(Image.open(ROOT / row["path"]))
        amb.append(dict(id=row["id"], path=row["path"], **result))
    Path("docs/results/ambiguous.json").write_text(json.dumps(amb, indent=2))


if __name__ == "__main__":
    evaluate()
