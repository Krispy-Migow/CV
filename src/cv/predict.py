"""Inferencia sobre una imagen: top-k razas más probables.

Uso:
    uv run python -m cv.predict ruta/a/imagen.jpg --topk 3
"""
import argparse
from pathlib import Path

import torch
from PIL import Image

from cv.config import load_config, resolve
from cv.data import build_transforms
from cv.model import load_checkpoint
from cv.species import species_of
from cv.train import get_device


@torch.no_grad()
def predict_image(image_path: str | Path, model, classes: list[str], image_size: int,
                  device: torch.device, topk: int = 3) -> list[tuple[str, float]]:
    img = Image.open(image_path).convert("RGB")
    x = build_transforms(image_size, train=False)(img).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    values, indices = probs.topk(topk)
    return [(classes[i], v.item()) for v, i in zip(values, indices)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predice la raza de una imagen.")
    parser.add_argument("image", help="Ruta a la imagen")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config()
    device = get_device()
    model, ckpt = load_checkpoint(resolve(cfg["train"]["checkpoint"]), device)
    results = predict_image(args.image, model, ckpt["classes"], ckpt["image_size"],
                            device, args.topk)
    for name, p in results:
        print(f"{name:<30} {p * 100:5.1f}%  ({species_of(name)})")


if __name__ == "__main__":
    main()
