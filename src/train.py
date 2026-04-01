import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data.dataset import get_datasets
from src.models.cnn import SimpleCNN
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def evaluate(model, loader, device, phase="Test"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.info(f"\n=== {phase} RESULTS ===")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-score:  {f1:.4f}\n")

    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def plot_losses(train_losses, val_losses, config_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='o')
    plt.title(f"Train vs Validation Loss — {config_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    Path("results").mkdir(exist_ok=True)
    plt.savefig(f"results/loss_plot_{config_name}.png")
    plt.close()
    logger.info(f"Loss plot saved: results/loss_plot_{config_name}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config_1batch.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        full_config = yaml.safe_load(f)

    setup_logging(full_config)
    logger.info(f"Running with config: {args.config}")

    train_loader, val_loader, test_loader, dataset_config = get_datasets(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=dataset_config["learning_rate"])

    train_losses = []
    val_losses = []

    for epoch in range(1, dataset_config["num_epochs"] + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train = running_loss / len(train_loader)
        train_losses.append(avg_train)

        # Validation loss
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            avg_val = val_loss / len(val_loader)
            val_losses.append(avg_val)
            logger.info(f"Epoch [{epoch}/{dataset_config['num_epochs']}] "
                        f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        else:
            logger.info(f"Epoch [{epoch}/{dataset_config['num_epochs']}] Train Loss: {avg_train:.4f}")

    metrics = evaluate(model, test_loader, device, "FINAL TEST (STATIC)")

    config_name = Path(args.config).stem
    plot_losses(train_losses, val_losses, config_name)

    Path("results").mkdir(exist_ok=True)
    with open(f"results/results_{config_name}.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config,
            "metrics": metrics,
            "train_batches": dataset_config["train_batch_ids"],
            "val_batches": dataset_config.get("val_batch_ids", [])
        }, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved: results/results_{config_name}.json")


if __name__ == "__main__":
    main()