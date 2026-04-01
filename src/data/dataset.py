import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import yaml
import logging

logger = logging.getLogger(__name__)


def load_cifar_batch(file_path: Path):
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    data = batch["data"]
    labels = batch["labels"]
    data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
    return data, np.array(labels)


class CustomCIFAR(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


def get_datasets(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)["dataset"]

    root = Path(config["root"]) / "cifar-10-batches-py"

    if not root.exists():
        logger.info("Downloading CIFAR-10...")
        _ = datasets.CIFAR10(root=str(root.parent), download=True)

    # Train (dynamically from selected batches)
    train_data, train_labels = [], []
    for bid in config["train_batch_ids"]:
        d, l = load_cifar_batch(root / f"data_batch_{bid}")
        train_data.append(d)
        train_labels.extend(l)
    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.array(train_labels)

    # Val (dynamically from selected batches)
    val_data, val_labels = [], []
    for bid in config.get("val_batch_ids", []):
        d, l = load_cifar_batch(root / f"data_batch_{bid}")
        val_data.append(d)
        val_labels.extend(l)
    val_data = np.concatenate(val_data, axis=0) if val_data else np.array([])
    val_labels = np.array(val_labels)

    # Test — static (always the same)
    test_data, test_labels = load_cifar_batch(root / "test_batch")

    logger.info(f"Train: {len(train_labels)} images (batches {config['train_batch_ids']})")
    logger.info(f"Val:   {len(val_labels)} images")
    logger.info(f"Test:  {len(test_labels)} images (STATIC)")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_ds = CustomCIFAR(train_data, train_labels, transform)
    val_ds = CustomCIFAR(val_data, val_labels, transform) if len(val_labels) > 0 else None
    test_ds = CustomCIFAR(test_data, test_labels, transform)

    bs = config["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, config