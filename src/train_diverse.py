"""Train the multi-format reader without replacing the original experiment artifacts."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .diverse import sample, family_name
from .learning import RobustCNN, preprocess, device
from .train import seed_all

CONFIG = dict(name="cnn_diverse", architecture="robust", crop=False, gray=True, size=96, normalize_background=True, dropout=.3)


def dataset(split, n):
    path = Path(f"data/diverse-v1/{split}-{n}.pt")
    if path.exists():
        return torch.load(path, weights_only=True)
    images, labels, families = [], [], []
    for i in range(n):
        im, label, meta = sample(i, split)
        images.append((preprocess(im, gray=True, size=96, normalize_background=True) * 255).byte())
        labels.append(label)
        families.append(meta["family"])
    data = dict(x=torch.stack(images), y=torch.tensor(labels), families=torch.tensor(families))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print(f"Prepared {split}: {n}", flush=True)
    return data


def evaluate(model, data, dev):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for x in data["x"].split(128):
            outputs.append(model(x.to(dev).float() / 255).cpu())
    logits = torch.cat(outputs)
    correct = logits.argmax(1) == data["y"]
    return dict(accuracy=correct.float().mean().item(), loss=nn.functional.cross_entropy(logits, data["y"]).item(), n=len(correct), per_family={family_name(int(f)): correct[data["families"] == f].float().mean().item() for f in data["families"].unique()}, per_class={str(i): correct[data["y"] == i].float().mean().item() for i in range(8)})


def train(epochs=25, n=16000):
    seed_all(731)
    start = time.perf_counter()
    dev = device()
    data = dataset("train", n)
    val = dataset("val", 1600)
    unseen = dataset("unseen_val", 800)
    model = RobustCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dl = DataLoader(TensorDataset(data["x"], data["y"]), batch_size=128, shuffle=True)
    best, history = float("inf"), []
    for epoch in range(epochs):
        model.train()
        total = count = 0
        for x, y in dl:
            x, y = x.to(dev).float() / 255, y.to(dev)
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            count += len(y)
        scheduler.step()
        v, u = evaluate(model, val, dev), evaluate(model, unseen, dev)
        row = dict(epoch=epoch + 1, train_loss=total / count, val_accuracy=v["accuracy"], unseen_val_accuracy=u["accuracy"], selection_loss=(v["loss"] + u["loss"]) / 2)
        history.append(row)
        print(row, flush=True)
        if row["selection_loss"] < best:
            best = row["selection_loss"]
            torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()}, config=CONFIG, epoch=epoch + 1), "models/cnn_diverse.pt")
    saved = torch.load("models/cnn_diverse.pt", weights_only=True)
    model.load_state_dict(saved["state"])
    result = dict(config=CONFIG, seed=731, selected_epoch=saved["epoch"], history=history, train_count=n, seconds=time.perf_counter() - start, device=str(dev))
    for split, count in [("val", 1600), ("unseen_val", 800), ("test", 1600), ("unseen_test", 800)]:
        result[split] = evaluate(model, dataset(split, count), dev)
    Path("docs/results/cnn_diverse.json").write_text(json.dumps(result, indent=2))
    print({s: result[s]["accuracy"] for s in ["val", "unseen_val", "test", "unseen_test"]}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--train-count", type=int, default=16000)
    args = p.parse_args()
    train(args.epochs, args.train_count)
