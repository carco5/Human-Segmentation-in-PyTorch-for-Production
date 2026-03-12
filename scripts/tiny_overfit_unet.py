import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import build_dataset_from_config
from src.models.unet import build_unet_from_config
from src.training.losses import build_loss_from_config
from src.training.metrics import (
    dice_score_from_logits,
    foreground_ratio,
    foreground_ratio_from_logits,
    iou_score_from_logits,
)
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

    losses = history["loss"]
    dice_scores = history["dice"]
    iou_scores = history["iou"]
    pred_fg = history["pred_fg_ratio"]
    target_fg = history["target_fg_ratio"]

    fig, axes = plt.subplots(4, 1, figsize=(8, 14))

    axes[0].plot(epochs, losses, marker="o")
    axes[0].set_title("Tiny Overfit Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    axes[1].plot(epochs, dice_scores, marker="o")
    axes[1].set_title("Tiny Overfit Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice")
    axes[1].grid(True)

    axes[2].plot(epochs, iou_scores, marker="o")
    axes[2].set_title("Tiny Overfit IoU")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("IoU")
    axes[2].grid(True)

    axes[3].plot(epochs, pred_fg, marker="o", label="predicted foreground ratio")
    axes[3].plot(epochs, target_fg, marker="o", label="target foreground ratio")
    axes[3].set_title("Foreground Ratio")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Ratio")
    axes[3].legend()
    axes[3].grid(True)

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

    tiny_config = config["debug"]["tiny_overfit"]
    num_samples = int(tiny_config["num_samples"])
    batch_size = int(tiny_config["batch_size"])
    num_epochs = int(tiny_config["num_epochs"])

    dataset = build_dataset_from_config(config, split="train")
    num_samples = min(num_samples, len(dataset))
    subset_indices = list(range(num_samples))
    subset = Subset(dataset, subset_indices)

    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = build_unet_from_config(config).to(device)
    criterion = build_loss_from_config(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )

    history = {
        "epochs": [],
        "loss": [],
        "dice": [],
        "iou": [],
        "pred_fg_ratio": [],
        "target_fg_ratio": [],
    }

    print("=" * 60)
    print("Tiny Overfit Test - U-Net")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Num samples: {num_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print()

    for epoch in range(1, num_epochs + 1):
        model.train()

        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_pred_fg = 0.0
        total_target_fg = 0.0
        batch_count = 0

        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dice += dice_score_from_logits(
                logits.detach(),
                masks.detach(),
                threshold=threshold,
                epsilon=epsilon,
            )
            total_iou += iou_score_from_logits(
                logits.detach(),
                masks.detach(),
                threshold=threshold,
                epsilon=epsilon,
            )
            total_pred_fg += foreground_ratio_from_logits(
                logits.detach(),
                threshold=threshold,
            )
            total_target_fg += foreground_ratio(masks.detach())

            batch_count += 1

        avg_loss = total_loss / batch_count
        avg_dice = total_dice / batch_count
        avg_iou = total_iou / batch_count
        avg_pred_fg = total_pred_fg / batch_count
        avg_target_fg = total_target_fg / batch_count

        history["epochs"].append(epoch)
        history["loss"].append(avg_loss)
        history["dice"].append(avg_dice)
        history["iou"].append(avg_iou)
        history["pred_fg_ratio"].append(avg_pred_fg)
        history["target_fg_ratio"].append(avg_target_fg)

        print(
            f"Epoch {epoch:02d}/{num_epochs} -> "
            f"loss: {avg_loss:.6f}, "
            f"dice: {avg_dice:.6f}, "
            f"iou: {avg_iou:.6f}, "
            f"pred_fg: {avg_pred_fg:.6f}, "
            f"target_fg: {avg_target_fg:.6f}"
        )

    metrics_output_path = Path(config["paths"]["metrics_dir"]) / "tiny_overfit_history.json"
    figures_output_path = Path(config["paths"]["figures_dir"]) / "tiny_overfit_curves.png"

    save_history(history, metrics_output_path)
    plot_history(history, figures_output_path)

    print("\nTiny overfit test finished.")
    print(f"Metrics saved to: {metrics_output_path}")
    print(f"Curves saved to:  {figures_output_path}")


if __name__ == "__main__":
    main()