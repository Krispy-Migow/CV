"""Entrenamiento: bucle train/val y guardado del mejor checkpoint.

Uso:
    uv run python -m cv.train --epochs 3
"""
import argparse
import json

import torch
from torch import nn
from tqdm import tqdm

from cv.config import ROOT, load_config, resolve
from cv.data import get_dataloaders
from cv.model import build_model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    """Una pasada completa. Si hay optimizer, entrena; si no, solo evalúa."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            n += x.size(0)

    return total_loss / n, correct / n


def train(epochs: int | None = None, cfg: dict | None = None) -> list[dict]:
    cfg = cfg or load_config()
    torch.manual_seed(cfg["seed"])
    device = get_device()
    epochs = epochs or cfg["train"]["epochs"]

    loaders = get_dataloaders(cfg)
    classes = loaders["classes"]

    model = build_model(
        num_classes=len(classes),
        freeze_backbone=cfg["model"]["freeze_backbone"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # Solo pasamos a Adam los parámetros que realmente se entrenan.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg["train"]["lr"])

    ckpt_path = resolve(cfg["train"]["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = resolve(cfg["evaluate"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dispositivo: {device} | clases: {len(classes)} | épocas: {epochs}")
    history, best_acc = [], 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, loaders["val"], criterion, None, device)
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        print(f"Época {epoch}/{epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.3f}")

        # Guardamos solo cuando mejora la validación (no usamos el test aquí).
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "image_size": cfg["data"]["image_size"]}, ckpt_path)

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Mejor val acc: {best_acc:.3f} | checkpoint: {ckpt_path.relative_to(ROOT)}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el clasificador de razas.")
    parser.add_argument("--epochs", type=int, default=None, help="Épocas (por defecto: config)")
    args = parser.parse_args()
    train(epochs=args.epochs)


if __name__ == "__main__":
    main()
