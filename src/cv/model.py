"""Definición del modelo y carga de checkpoints."""
from pathlib import Path

import torch
from torch import nn
from torchvision import models


def build_model(num_classes: int = 37, freeze_backbone: bool = True,
                pretrained: bool = True) -> nn.Module:
    """ResNet18 con una nueva capa final de `num_classes` salidas.

    freeze_backbone=True congela todas las capas previas: solo se entrena `fc`.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    # La nueva capa fc nace con requires_grad=True, así que es la única que aprende.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Reconstruye el modelo entrenado a partir de un checkpoint guardado por train.py."""
    ckpt = torch.load(path, map_location=device)
    model = build_model(num_classes=len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
