"""One prediction path for evaluation and UI. Codex-generated; see ATTRIBUTION.md."""

import json
from pathlib import Path
import joblib
import numpy as np
import torch
from .learning import SmallCNN, backbone, preprocess, imagenet_input
from .payoffs import CLASSES, EXPLANATIONS


class Predictor:
    def __init__(self, name=None):
        torch.set_num_threads(4)
        if name is None:
            name = json.loads(Path("models/selection.json").read_text())["name"]
        self.name = name
        self.config = json.loads(Path(f"docs/results/{name}.json").read_text())[
            "config"
        ]
        if name == "baseline":
            self.model = joblib.load("models/baseline.joblib")
        elif name == "transfer":
            self.model = joblib.load("models/transfer.joblib")
            self.net = backbone()
            self.net.load_state_dict(
                torch.load(
                    "models/transfer_backbone.pt", map_location="cpu", weights_only=True
                )
            )
            self.net.eval()
        else:
            saved = torch.load(
                f"models/{name}.pt", map_location="cpu", weights_only=True
            )
            self.model = SmallCNN(saved["config"]["dropout"])
            self.model.load_state_dict(saved["state"])
            self.model.eval()

    def probabilities(self, x):
        with torch.inference_mode():
            if self.name == "baseline":
                return self.model.predict_proba(x.flatten(1).numpy())
            if self.name == "transfer":
                return self.model.predict_proba(self.net(imagenet_input(x)).numpy())
            return self.model(x).softmax(1).numpy()

    def prepare(self, im):
        c = self.config
        return preprocess(
            im, c.get("crop", False), c.get("gray", False), c.get("size", 64)
        )

    def predict(self, im):
        p = self.probabilities(self.prepare(im)[None])[0]
        label = int(p.argmax())
        return dict(
            category=CLASSES[label],
            confidence=float(p[label]),
            explanation=EXPLANATIONS[label],
            model=self.name,
            probabilities={c: float(v) for c, v in zip(CLASSES, p)},
        )
