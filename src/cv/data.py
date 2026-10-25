"""Datasets, transformaciones y DataLoaders para Oxford-IIIT Pet."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from cv.config import load_config, resolve

# Medias y desviaciones de ImageNet: el modelo preentrenado las espera.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int = 224, train: bool = True) -> transforms.Compose:
    """Transformaciones de entrenamiento (con augmentation) o de evaluación."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])


def get_dataloaders(cfg: dict | None = None) -> dict:
    """Devuelve train/val/test DataLoaders y la lista de nombres de clases."""
    cfg = cfg or load_config()
    root = resolve(cfg["data"]["root"])
    size = cfg["data"]["image_size"]
    batch = cfg["train"]["batch_size"]
    workers = cfg["data"]["num_workers"]

    # "trainval" se parte en train/val. Usamos dos datasets con transformaciones
    # distintas (augmentation solo en train) y los mismos índices seleccionados
    # con una semilla fija, para que la partición sea reproducible.
    train_full = datasets.OxfordIIITPet(
        root, split="trainval", target_types="category", download=True,
        transform=build_transforms(size, train=True),
    )
    val_full = datasets.OxfordIIITPet(
        root, split="trainval", target_types="category",
        transform=build_transforms(size, train=False),
    )
    test = datasets.OxfordIIITPet(
        root, split="test", target_types="category",
        transform=build_transforms(size, train=False),
    )

    gen = torch.Generator().manual_seed(cfg["seed"])
    perm = torch.randperm(len(train_full), generator=gen).tolist()
    n_val = int(len(perm) * cfg["data"]["val_split"])
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    loaders = {
        "train": DataLoader(Subset(train_full, train_idx), batch_size=batch,
                            shuffle=True, num_workers=workers),
        "val": DataLoader(Subset(val_full, val_idx), batch_size=batch,
                          num_workers=workers),
        "test": DataLoader(test, batch_size=batch, num_workers=workers),
        "classes": train_full.classes,
    }
    return loaders
