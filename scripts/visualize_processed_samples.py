from pathlib import Path

import matplotlib.pyplot as plt
import torch

from src.data.dataset import build_dataset_from_config
from src.utils.config import load_config


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


def plot_split_samples(config: dict, split: str, max_samples: int = 3) -> Path:
    dataset = build_dataset_from_config(config, split=split)
    sample_count = min(len(dataset), max_samples)

    figure, axes = plt.subplots(sample_count, 3, figsize=(12, 4 * sample_count))

    if sample_count == 1:
        axes = [axes]

    for row_idx in range(sample_count):
        sample = dataset[row_idx]
        image = tensor_to_image(sample["image"])
        mask = tensor_to_mask(sample["mask"])
        overlay = make_overlay(image, mask)

        axes[row_idx][0].imshow(image)
        axes[row_idx][0].set_title(f"{split} image #{row_idx}")
        axes[row_idx][0].axis("off")

        axes[row_idx][1].imshow(mask, cmap="gray")
        axes[row_idx][1].set_title(f"{split} mask #{row_idx}")
        axes[row_idx][1].axis("off")

        axes[row_idx][2].imshow(overlay)
        axes[row_idx][2].set_title(f"{split} overlay #{row_idx}")
        axes[row_idx][2].axis("off")

    figure.tight_layout()

    output_dir = Path(config["paths"]["figures_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{split}_dataset_preview.png"
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return output_path


def main() -> None:
    config = load_config("configs/base.yaml")

    print("=" * 60)
    print("Processed Dataset Visualization")
    print("=" * 60)

    for split in ["train", "val"]:
        output_path = plot_split_samples(config, split=split, max_samples=3)
        print(f"[OK] Saved preview for '{split}' split to: {output_path}")


if __name__ == "__main__":
    main()