import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.data.dataset import build_dataset_from_config
from src.inference.predict import get_checkpoint_path, load_model_from_checkpoint, predict_from_image_tensor
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize checkpoint predictions.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def tensor_to_image(image_tensor: torch.Tensor):
    return image_tensor.permute(1, 2, 0).cpu().numpy()


def tensor_to_mask(mask_tensor: torch.Tensor):
    return mask_tensor.squeeze(0).cpu().numpy()


def make_overlay(image, mask):
    overlay = image.copy()
    overlay[..., 0] = overlay[..., 0] * (1 - 0.4 * mask) + 1.0 * (0.4 * mask)
    overlay[..., 1] = overlay[..., 1] * (1 - 0.4 * mask)
    overlay[..., 2] = overlay[..., 2] * (1 - 0.4 * mask)
    return overlay.clip(0.0, 1.0)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    device = torch.device(config["training"]["device"])
    threshold = float(config["evaluation"]["threshold"])

    checkpoint_path = get_checkpoint_path(config)
    model, checkpoint = load_model_from_checkpoint(config, device=device, checkpoint_path=checkpoint_path)

    dataset = build_dataset_from_config(config, split="val")
    num_samples = min(3, len(dataset))

    figure, axes = plt.subplots(num_samples, 5, figsize=(18, 5 * num_samples))

    if num_samples == 1:
        axes = [axes]

    for row_idx in range(num_samples):
        sample = dataset[row_idx]

        image_tensor = sample["image"].to(device)
        target_mask_tensor = sample["mask"]

        outputs = predict_from_image_tensor(
            model=model,
            image_tensor=image_tensor.to(device),
            threshold=threshold,
        )

        image = tensor_to_image(sample["image"])
        target_mask = tensor_to_mask(target_mask_tensor)
        probability_map = tensor_to_mask(outputs["probabilities"].squeeze(0).cpu())
        predicted_mask = tensor_to_mask(outputs["predictions"].squeeze(0).cpu())
        overlay = make_overlay(image, predicted_mask)

        axes[row_idx][0].imshow(image)
        axes[row_idx][0].set_title(f"Image #{row_idx}")
        axes[row_idx][0].axis("off")

        axes[row_idx][1].imshow(target_mask, cmap="gray")
        axes[row_idx][1].set_title("Ground Truth")
        axes[row_idx][1].axis("off")

        axes[row_idx][2].imshow(probability_map, cmap="viridis")
        axes[row_idx][2].set_title("Predicted Probability")
        axes[row_idx][2].axis("off")

        axes[row_idx][3].imshow(predicted_mask, cmap="gray")
        axes[row_idx][3].set_title("Predicted Mask")
        axes[row_idx][3].axis("off")

        axes[row_idx][4].imshow(overlay)
        axes[row_idx][4].set_title("Prediction Overlay")
        axes[row_idx][4].axis("off")

    figure.tight_layout()

    output_dir = Path(config["paths"]["predictions_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = config["experiment"]["name"]
    output_path = output_dir / f"{experiment_name}_val_predictions.png"

    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    print("=" * 60)
    print("Checkpoint Prediction Visualization")
    print("=" * 60)
    print(f"Config path: {args.config}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint metric value: {checkpoint['metric_value']:.6f}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()