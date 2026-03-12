import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataloader import build_dataloader_from_config
from src.models.unet import build_unet_from_config
from src.training.engine import train_one_epoch, validate_one_epoch
from src.training.losses import build_loss_from_config
from src.utils.config import load_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_history(history: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def plot_history(history: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = history["epochs"]

    train_loss = [epoch_data["loss"] for epoch_data in history["train"]]
    val_loss = [epoch_data["loss"] for epoch_data in history["val"]]

    train_dice = [epoch_data["dice"] for epoch_data in history["train"]]
    val_dice = [epoch_data["dice"] for epoch_data in history["val"]]

    train_iou = [epoch_data["iou"] for epoch_data in history["train"]]
    val_iou = [epoch_data["iou"] for epoch_data in history["val"]]

    fig, axes = plt.subplots(3, 1, figsize=(8, 12))

    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_dice, marker="o", label="train")
    axes[1].plot(epochs, val_dice, marker="o", label="val")
    axes[1].set_title("Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, train_iou, marker="o", label="train")
    axes[2].plot(epochs, val_iou, marker="o", label="val")
    axes[2].set_title("IoU")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    config = load_config("configs/base.yaml")

    seed = int(config["project"]["seed"])
    set_seed(seed)

    device = torch.device(config["training"]["device"])
    threshold = float(config["evaluation"]["threshold"])
    epsilon = float(config["evaluation"]["epsilon"])

    model = build_unet_from_config(config).to(device)
    criterion = build_loss_from_config(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )

    train_loader = build_dataloader_from_config(config, split="train", shuffle=True)
    val_loader = build_dataloader_from_config(config, split="val", shuffle=False)

    num_epochs = int(config["training"]["num_epochs"])
    max_train_batches = int(config["training"]["max_train_batches"])
    max_val_batches = int(config["training"]["max_val_batches"])

    history = {
        "epochs": [],
        "train": [],
        "val": [],
    }

    print("=" * 60)
    print("U-Net Baseline Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Max train batches per epoch: {max_train_batches}")
    print(f"Max val batches per epoch: {max_val_batches}")
    print()

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            threshold=threshold,
            epsilon=epsilon,
            max_batches=max_train_batches,
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            epsilon=epsilon,
            max_batches=max_val_batches,
        )

        history["epochs"].append(epoch)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        print(
            f"Train -> loss: {train_metrics['loss']:.6f}, "
            f"dice: {train_metrics['dice']:.6f}, "
            f"iou: {train_metrics['iou']:.6f}"
        )
        print(
            f"Val   -> loss: {val_metrics['loss']:.6f}, "
            f"dice: {val_metrics['dice']:.6f}, "
            f"iou: {val_metrics['iou']:.6f}"
        )
        print("-" * 60)

    metrics_output_path = Path(config["paths"]["metrics_dir"]) / "unet_baseline_history.json"
    figures_output_path = Path(config["paths"]["figures_dir"]) / "unet_baseline_curves.png"

    save_history(history, metrics_output_path)
    plot_history(history, figures_output_path)

    print("\nTraining finished successfully.")
    print(f"Metrics saved to: {metrics_output_path}")
    print(f"Curves saved to:  {figures_output_path}")


if __name__ == "__main__":
    main()