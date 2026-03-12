import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net baseline for binary human segmentation.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Optional override for number of epochs.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional override for max train batches per epoch.",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=None,
        help="Optional override for max validation batches per epoch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional override for device (e.g. cpu, cuda).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_history(history: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric_value: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric_value": metric_value,
        },
        output_path,
    )


def plot_history(history: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = history["epochs"]

    train_loss = [epoch_data["loss"] for epoch_data in history["train"]]
    val_loss = [epoch_data["loss"] for epoch_data in history["val"]]

    train_dice = [epoch_data["dice"] for epoch_data in history["train"]]
    val_dice = [epoch_data["dice"] for epoch_data in history["val"]]

    train_iou = [epoch_data["iou"] for epoch_data in history["train"]]
    val_iou = [epoch_data["iou"] for epoch_data in history["val"]]

    train_pred_fg = [epoch_data["pred_fg_ratio"] for epoch_data in history["train"]]
    val_pred_fg = [epoch_data["pred_fg_ratio"] for epoch_data in history["val"]]

    train_target_fg = [epoch_data["target_fg_ratio"] for epoch_data in history["train"]]
    val_target_fg = [epoch_data["target_fg_ratio"] for epoch_data in history["val"]]

    fig, axes = plt.subplots(4, 1, figsize=(8, 16))

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

    axes[3].plot(epochs, train_pred_fg, marker="o", label="train predicted fg")
    axes[3].plot(epochs, val_pred_fg, marker="o", label="val predicted fg")
    axes[3].plot(epochs, train_target_fg, marker="o", linestyle="--", label="train target fg")
    axes[3].plot(epochs, val_target_fg, marker="o", linestyle="--", label="val target fg")
    axes[3].set_title("Foreground Ratio")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Ratio")
    axes[3].legend()
    axes[3].grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def resolve_optional_int(value):
    if value is None:
        return None
    return int(value)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config["project"]["seed"])
    set_seed(seed)

    device_name = args.device if args.device is not None else config["training"]["device"]
    device = torch.device(device_name)

    threshold = float(config["evaluation"]["threshold"])
    epsilon = float(config["evaluation"]["epsilon"])

    experiment_name = config["experiment"]["name"]
    checkpoint_metric = config["experiment"]["checkpoint_metric"]

    model = build_unet_from_config(config).to(device)
    criterion = build_loss_from_config(config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
    )

    train_loader = build_dataloader_from_config(config, split="train", shuffle=True)
    val_loader = build_dataloader_from_config(config, split="val", shuffle=False)

    num_epochs = int(args.num_epochs) if args.num_epochs is not None else int(config["training"]["num_epochs"])
    max_train_batches = (
        int(args.max_train_batches)
        if args.max_train_batches is not None
        else resolve_optional_int(config["training"]["max_train_batches"])
    )
    max_val_batches = (
        int(args.max_val_batches)
        if args.max_val_batches is not None
        else resolve_optional_int(config["training"]["max_val_batches"])
    )

    history = {
        "experiment_name": experiment_name,
        "config_path": args.config,
        "epochs": [],
        "train": [],
        "val": [],
        "best_epoch": None,
        "best_val_metric": None,
        "checkpoint_metric": checkpoint_metric,
    }

    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])

    checkpoint_path = checkpoints_dir / f"{experiment_name}_best.pt"
    metrics_output_path = metrics_dir / f"{experiment_name}_history.json"
    figures_output_path = figures_dir / f"{experiment_name}_curves.png"

    best_val_metric = float("-inf")

    print("=" * 60)
    print("U-Net Baseline Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Experiment: {experiment_name}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Max train batches per epoch: {max_train_batches}")
    print(f"Max val batches per epoch: {max_val_batches}")
    print(f"Checkpoint metric: val_{checkpoint_metric}")
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

        current_val_metric = float(val_metrics[checkpoint_metric])

        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            history["best_epoch"] = epoch
            history["best_val_metric"] = best_val_metric

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric_value=best_val_metric,
                output_path=checkpoint_path,
            )
            print(f"[OK] New best checkpoint saved at epoch {epoch}")

        print(
            f"Train -> loss: {train_metrics['loss']:.6f}, "
            f"dice: {train_metrics['dice']:.6f}, "
            f"iou: {train_metrics['iou']:.6f}, "
            f"pred_fg: {train_metrics['pred_fg_ratio']:.6f}, "
            f"target_fg: {train_metrics['target_fg_ratio']:.6f}"
        )
        print(
            f"Val   -> loss: {val_metrics['loss']:.6f}, "
            f"dice: {val_metrics['dice']:.6f}, "
            f"iou: {val_metrics['iou']:.6f}, "
            f"pred_fg: {val_metrics['pred_fg_ratio']:.6f}, "
            f"target_fg: {val_metrics['target_fg_ratio']:.6f}"
        )
        print("-" * 60)

    save_history(history, metrics_output_path)
    plot_history(history, figures_output_path)

    print("\nTraining finished successfully.")
    print(f"Best epoch: {history['best_epoch']}")
    print(f"Best val {checkpoint_metric}: {history['best_val_metric']:.6f}")
    print(f"Checkpoint saved to: {checkpoint_path}")
    print(f"Metrics saved to:    {metrics_output_path}")
    print(f"Curves saved to:     {figures_output_path}")


if __name__ == "__main__":
    main()