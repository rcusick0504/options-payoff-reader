"""Build figures and comparison tables from saved results. Codex-generated."""

import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from .payoffs import CLASSES
from .generate import ROOT
from .learning import records


def report():
    root = Path("docs/results")
    ev = json.loads((root / "evaluation.json").read_text())
    sel = json.loads(Path("models/selection.json").read_text())
    name = sel["name"]
    rows = [
        "# Experiment results",
        "",
        f"Selected model: **{name}**, chosen on validation before held-out evaluation. Results below are from actual runs, seed 386.",
        "",
        "## Model comparison",
        "",
        "| Run | Val accuracy | Test accuracy | Macro-F1 | Log loss | OOD accuracy | Independent renderer | Size MB | Median ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, e in ev.items():
        r = json.loads((root / f"{n}.json").read_text())
        t = e["test"]
        l = e["latency"]
        rows.append(
            f"| {n} | {r['val_accuracy']:.4f} | {t['accuracy']:.4f} | {t['macro_f1']:.4f} | {t['log_loss']:.6f} | {e['ood']['accuracy']:.4f} | {e['independent']['accuracy']:.4f} | {l['model_bytes']/1e6:.3f} | {l['median_ms']:.2f} |"
        )
    rows += [
        "",
        "Accuracy and balanced accuracy are identical here because classes have equal support. Log loss provides a distinct probability-sensitive metric. All latency results use CPU, batch one, four torch threads, 10 warmups, 50 timed calls; preprocessing included, decoding/loading excluded. Full per-class metrics, p95 and platform are in results/evaluation.json.",
        "",
        "## Hyperparameter tuning",
        "",
        "| Run | Learning rate | Dropout | Selected epoch | Epochs run | Validation loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for n in ["cnn_reference", "cnn_lr_low", "cnn_dropout"]:
        r = json.loads((root / f"{n}.json").read_text())
        c = r["config"]
        rows.append(
            f"| {n} | {c['lr']} | {c['dropout']} | {r['selected_epoch']} | {len(r['history'])} | {r['val_loss']:.6f} |"
        )
    rows += [
        "",
        "## Augmentation impact",
        "",
        "| Run | Validation accuracy | Test accuracy | OOD accuracy |",
        "|---|---:|---:|---:|",
    ]
    for n in ["cnn_reference", "cnn_augmented"]:
        r = json.loads((root / f"{n}.json").read_text())
        rows.append(
            f"| {n} | {r['val_accuracy']:.4f} | {ev[n]['test']['accuracy']:.4f} | {ev[n]['ood']['accuracy']:.4f} |"
        )
    delta = (
        ev["cnn_augmented"]["ood"]["accuracy"] - ev["cnn_reference"]["ood"]["accuracy"]
    )
    rows += [
        "",
        f"Brightness/contrast augmentation changed OOD accuracy by {delta*100:+.2f} percentage points in this single-seed comparison. A change is evidence of impact, not proof of a repeatable improvement. No held-out result was used to retune.",
        "",
        "## Two by two preprocessing ablation",
        "",
        "| Input region | Channels | Run | Val accuracy | Test accuracy | OOD accuracy |",
        "|---|---|---|---:|---:|---:|",
    ]
    for n, region, color in [
        ("cnn_reference", "Full", "Color"),
        ("cnn_full_gray", "Full", "Grayscale"),
        ("cnn_crop_color", "Plot box", "Color"),
        ("cnn_crop_gray", "Plot box", "Grayscale"),
    ]:
        r = json.loads((root / f"{n}.json").read_text())
        rows.append(
            f"| {region} | {color} | {n} | {r['val_accuracy']:.4f} | {ev[n]['test']['accuracy']:.4f} | {ev[n]['ood']['accuracy']:.4f} |"
        )
    rows += [
        "",
        "These four runs use the reference learning rate/dropout, seed and maximum epoch budget, with no augmentation. Early stopping can change actual epochs. The fixed crop removes peripheral labels but keeps axes/grid; this is not learned plot detection. The same reference run serves as the explicitly shared control.",
        "",
        "## Selection and technical tradeoff",
        "",
        f"The predeclared rule selected {name} at validation accuracy {sel['val_accuracy']:.4f}. Among tied validation scores, checkpoint size broke the tie. It uses {ev[name]['latency']['model_bytes']/1e6:.3f} MB versus {ev['baseline']['latency']['model_bytes']/1e6:.3f} MB for the stored nearest-neighbor baseline. Measured median inference is {ev[name]['latency']['median_ms']:.2f} ms versus {ev['baseline']['latency']['median_ms']:.2f} ms for that baseline. The baseline stores training pixels and searches them; a CNN stores learned weights. These are concrete storage/latency tradeoffs, not a claim that neural models are always necessary.",
        "",
        f"The selected model's standard test accuracy is {ev[name]['test']['accuracy']:.2%}, unfamiliar-style accuracy is {ev[name]['ood']['accuracy']:.2%}, and independent-renderer accuracy is {ev[name]['independent']['accuracy']:.2%}. These synthetic results do not establish performance on real screenshots. The transfer comparison uses ImageNet pretraining and 96px inputs; the CNN uses 64px. Equal labels/splits are controlled, but pretraining and resolution differ.",
        "",
        "## Training curves",
        "",
        "![Training and validation histories](figures/training_curves.png)",
        "",
        "## Confusion matrices",
        "",
        "![Selected model confusion matrices](figures/confusion.png)",
    ]
    Path("docs/RESULTS.md").write_text("\n".join(rows) + "\n")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flat
    for ax, n in zip(axes, [n for n in ev if n.startswith("cnn")]):
        h = json.loads((root / f"{n}.json").read_text())["history"]
        ep = [r["epoch"] for r in h]
        ax.plot(ep, [r["train_loss"] for r in h], label="train loss")
        ax.plot(ep, [r["val_loss"] for r in h], label="val loss")
        ax.set_title(n)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    axes[7].axis("off")
    fig.tight_layout()
    fig.savefig("docs/figures/training_curves.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, split in zip(axes, ["test", "ood", "independent"]):
        cm = np.array(ev[name][split]["confusion"])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f'{split}: {ev[name][split]["accuracy"]:.1%}')
        ax.set_xticks(range(8), CLASSES, rotation=65, ha="right", fontsize=8)
        ax.set_yticks(range(8), CLASSES, fontsize=8)
        ax.set(xlabel="Predicted", ylabel="True")
        for i in range(8):
            for j in range(8):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )
    fig.tight_layout()
    fig.savefig("docs/figures/confusion.png", dpi=140)
    plt.close(fig)
    errors = []
    for split in ["test", "ood", "independent"]:
        predictions = json.loads((root / f"predictions_{split}.json").read_text())
        bad = [r for r in predictions if r["truth"] != r["prediction"]]
        errors.extend(sorted(bad, key=lambda r: -r["confidence"])[:4])
    shown = errors[:8]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, shown):
        ax.imshow(Image.open(ROOT / r["path"]))
        ax.set_title(
            f"{r['id']}\nTrue: {CLASSES[r['truth']]}\nPred: {CLASSES[r['prediction']]} ({r['confidence']:.0%})",
            fontsize=9,
        )
    if not shown:
        fig.suptitle(
            "No errors in these evaluated sets; do not infer real-world reliability."
        )
    fig.tight_layout()
    fig.savefig("docs/figures/failures.png", dpi=140)
    plt.close(fig)
    ambiguous = json.loads((root / "ambiguous.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    for ax, r in zip(axes, sorted(ambiguous, key=lambda r: -r["confidence"])[:4]):
        ax.imshow(Image.open(ROOT / r["path"]))
        ax.set_title(
            f"Incomplete chart\n{r['category']} ({r['confidence']:.0%})", fontsize=9
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("docs/figures/ambiguous.png", dpi=140)
    plt.close(fig)
    groups = []
    preds = {
        r["id"]: r for r in json.loads((root / "predictions_ood.json").read_text())
    }
    for bg in sorted({r["style"]["background"] for r in records("ood")}):
        subset = [
            preds[r["id"]] for r in records("ood") if r["style"]["background"] == bg
        ]
        groups.append(
            dict(
                background=bg,
                n=len(subset),
                accuracy=float(
                    np.mean([r["truth"] == r["prediction"] for r in subset])
                ),
            )
        )
    Path("docs/results/ood_groups.json").write_text(json.dumps(groups, indent=2))
    print("Reports and figures generated", flush=True)


if __name__ == "__main__":
    report()
