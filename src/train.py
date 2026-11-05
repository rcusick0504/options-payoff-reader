"""Validation-only experiment runner. AI-generated with Codex; see ATTRIBUTION.md."""

import argparse
import json
import platform
import random
import time
from pathlib import Path
import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from .learning import SmallCNN, load_split, device, augment, backbone, imagenet_input

RESULTS = Path("docs/results")
RESULTS.mkdir(parents=True, exist_ok=True)
CONFIGS = [
    dict(
        name="cnn_reference",
        lr=0.003,
        dropout=0.2,
        crop=False,
        gray=False,
        augmentation=False,
    ),
    dict(
        name="cnn_lr_low",
        lr=0.001,
        dropout=0.2,
        crop=False,
        gray=False,
        augmentation=False,
    ),
    dict(
        name="cnn_dropout",
        lr=0.003,
        dropout=0.4,
        crop=False,
        gray=False,
        augmentation=False,
    ),
    dict(
        name="cnn_augmented",
        lr=0.003,
        dropout=0.2,
        crop=False,
        gray=False,
        augmentation=True,
    ),
    dict(
        name="cnn_full_gray",
        lr=0.003,
        dropout=0.2,
        crop=False,
        gray=True,
        augmentation=False,
    ),
    dict(
        name="cnn_crop_color",
        lr=0.003,
        dropout=0.2,
        crop=True,
        gray=False,
        augmentation=False,
    ),
    dict(
        name="cnn_crop_gray",
        lr=0.003,
        dropout=0.2,
        crop=True,
        gray=True,
        augmentation=False,
    ),
]


def seed_all(seed=386):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)


def save_result(name, result):
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2))


def cnn(cfg, epochs=12):
    seed_all()
    dev = device()
    start = time.perf_counter()
    x, y, _ = load_split("train", cfg["crop"], cfg["gray"])
    vx, vy, _ = load_split("val", cfg["crop"], cfg["gray"])
    dl = DataLoader(
        TensorDataset(x, y),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(386),
    )
    vl = DataLoader(TensorDataset(vx, vy), batch_size=256)
    model = SmallCNN(cfg["dropout"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()
    best = float("inf")
    stale = 0
    history = []
    for epoch in range(epochs):
        model.train()
        total = correct = count = 0
        for bx, by in dl:
            bx, by = bx.to(dev), by.to(dev)
            if cfg["augmentation"]:
                bx = augment(bx)
            opt.zero_grad()
            out = model(bx)
            loss = loss_fn(out, by)
            loss.backward()
            opt.step()
            total += loss.item() * len(by)
            correct += (out.argmax(1) == by).sum().item()
            count += len(by)
        model.eval()
        vt = vc = vn = 0
        with torch.inference_mode():
            for bx, by in vl:
                bx, by = bx.to(dev), by.to(dev)
                out = model(bx)
                vt += loss_fn(out, by).item() * len(by)
                vc += (out.argmax(1) == by).sum().item()
                vn += len(by)
        row = dict(
            epoch=epoch + 1,
            train_loss=total / count,
            train_accuracy=correct / count,
            val_loss=vt / vn,
            val_accuracy=vc / vn,
        )
        history.append(row)
        print(cfg["name"], row, flush=True)
        if row["val_loss"] < best - 0.0001:
            best = row["val_loss"]
            stale = 0
            # Copy tensors for serialization without moving the live model/optimizer.
            state = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            torch.save(
                dict(state=state, config=cfg, epoch=epoch + 1),
                f"models/{cfg['name']}.pt",
            )
            selected = row
        else:
            stale += 1
        if stale >= 3:
            break
    result = dict(
        config=cfg,
        seed=386,
        batch_size=128,
        max_epochs=epochs,
        patience=3,
        device=str(dev),
        hardware=platform.platform(),
        seconds=time.perf_counter() - start,
        selected_epoch=selected["epoch"],
        val_accuracy=selected["val_accuracy"],
        val_loss=selected["val_loss"],
        history=history,
        model_bytes=Path(f"models/{cfg['name']}.pt").stat().st_size,
    )
    save_result(cfg["name"], result)


def baseline():
    seed_all()
    start = time.perf_counter()
    x, y, _ = load_split("train", size=16)
    vx, vy, _ = load_split("val", size=16)
    model = KNeighborsClassifier(n_neighbors=3, weights="distance", n_jobs=4).fit(
        x.flatten(1).numpy(), y.numpy()
    )
    pred = model.predict(vx.flatten(1).numpy())
    joblib.dump(model, "models/baseline.joblib")
    save_result(
        "baseline",
        dict(
            config=dict(name="baseline", size=16, crop=False, gray=False),
            val_accuracy=float(np.mean(pred == vy.numpy())),
            seconds=time.perf_counter() - start,
            model_bytes=Path("models/baseline.joblib").stat().st_size,
        ),
    )
    print("baseline complete", flush=True)


def features(net, x, dev):
    out = []
    with torch.inference_mode():
        for bx in x.split(128):
            out.append(net(imagenet_input(bx.to(dev))).cpu().numpy())
    return np.concatenate(out)


def transfer():
    seed_all()
    start = time.perf_counter()
    dev = device()
    torch.hub.set_dir("models/cache")
    net = backbone(pretrained=True).to(dev)
    x, y, _ = load_split("train", size=96)
    vx, vy, _ = load_split("val", size=96)
    f = features(net, x, dev)
    vf = features(net, vx, dev)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(f, y.numpy())
    joblib.dump(clf, "models/transfer.joblib")
    torch.save(net.cpu().state_dict(), "models/transfer_backbone.pt")
    save_result(
        "transfer",
        dict(
            config=dict(
                name="transfer",
                size=96,
                crop=False,
                gray=False,
                C=1.0,
                weights="MobileNet_V3_Small_Weights.IMAGENET1K_V1",
            ),
            val_accuracy=float(clf.score(vf, vy.numpy())),
            seconds=time.perf_counter() - start,
            model_bytes=Path("models/transfer.joblib").stat().st_size
            + Path("models/transfer_backbone.pt").stat().st_size,
            device=str(dev),
        ),
    )
    print("transfer complete", flush=True)


def select():
    names = ["baseline"] + [c["name"] for c in CONFIGS] + ["transfer"]
    results = [json.loads((RESULTS / f"{n}.json").read_text()) for n in names]
    winner = sorted(results, key=lambda r: (-r["val_accuracy"], r["model_bytes"]))[0]
    selection = dict(
        name=winner["config"]["name"],
        config=winner["config"],
        val_accuracy=winner["val_accuracy"],
        rule="Highest validation accuracy at validation-loss-selected checkpoint; smallest file size breaks ties.",
        candidates=[
            dict(
                name=r["config"]["name"],
                val_accuracy=r["val_accuracy"],
                model_bytes=r["model_bytes"],
            )
            for r in results
        ],
        test_used=False,
    )
    Path("models/selection.json").write_text(json.dumps(selection, indent=2))
    print(selection, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["baseline", "cnn", "transfer", "select"])
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument(
        "--rerun",
        action="store_true",
        help="Retrain even when a result exists; overwrites corresponding CNN artifacts",
    )
    a = p.parse_args()
    if a.stage == "cnn":
        for cfg in CONFIGS:
            if (
                a.rerun
                or not (RESULTS / f"{cfg['name']}.json").exists()
                or not Path(f"models/{cfg['name']}.pt").exists()
            ):
                cnn(cfg, a.epochs)
    else:
        globals()[a.stage]()
