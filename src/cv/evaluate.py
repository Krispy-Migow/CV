"""Evaluación sobre el conjunto de test: accuracy, reporte y matriz de confusión.

Uso:
    uv run python -m cv.evaluate
"""
import json

import matplotlib
matplotlib.use("Agg")  # no abre ventanas; guarda la figura en disco
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cv.config import load_config, resolve
from cv.data import get_dataloaders
from cv.model import load_checkpoint
from cv.train import get_device


@torch.no_grad()
def predict_all(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    for x, y in loader:
        out = model(x.to(device))
        y_pred.append(out.argmax(dim=1).cpu().numpy())
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def plot_confusion(cm: np.ndarray, classes: list[str], path) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión (test)")
    fig.colorbar(ax.images[0], ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    device = get_device()
    ckpt_path = resolve(cfg["train"]["checkpoint"])
    out_dir = resolve(cfg["evaluate"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_checkpoint(ckpt_path, device)
    classes = ckpt["classes"]
    loaders = get_dataloaders(cfg)

    y_true, y_pred = predict_all(model, loaders["test"], device)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes,
                                   output_dict=True, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plot_confusion(cm, classes, out_dir / "confusion_matrix.png")

    metrics = {"test_accuracy": acc, "macro_f1": report["macro avg"]["f1-score"],
               "per_class": {c: report[c] for c in classes}}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Test accuracy: {acc:.3f} | macro F1: {metrics['macro_f1']:.3f}")
    print(f"Guardado en {out_dir}")
    return metrics


if __name__ == "__main__":
    evaluate()
