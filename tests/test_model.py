"""Pruebas rápidas del modelo. No descargan el dataset."""
import torch

from cv.model import build_model


def test_output_shape_matches_num_classes():
    model = build_model(num_classes=37, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 37)


def test_frozen_backbone_only_head_trains():
    model = build_model(num_classes=37, freeze_backbone=True, pretrained=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}
