from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.config import load_config


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def create_sample_image_and_mask(
    image_size: int,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    # Background gradient
    x_gradient = np.linspace(30, 180, image_size, dtype=np.uint8)
    y_gradient = np.linspace(20, 120, image_size, dtype=np.uint8)

    image[..., 0] = x_gradient[np.newaxis, :]
    image[..., 1] = y_gradient[:, np.newaxis]
    image[..., 2] = 90

    # Synthetic "person" rectangle
    center_x = image_size // 2 + (sample_index % 3 - 1) * 10
    top = image_size // 4
    bottom = image_size - image_size // 8
    left = max(center_x - image_size // 8, 0)
    right = min(center_x + image_size // 8, image_size)

    mask[top:bottom, left:right] = 255

    # Color the foreground region differently in the RGB image
    image[top:bottom, left:right, 0] = 220
    image[top:bottom, left:right, 1] = 80
    image[top:bottom, left:right, 2] = 80

    return image, mask


def save_sample(image: np.ndarray, mask: np.ndarray, images_dir: Path, masks_dir: Path, name: str) -> None:
    Image.fromarray(image).save(images_dir / f"{name}.png")
    Image.fromarray(mask).save(masks_dir / f"{name}.png")


def main() -> None:
    config = load_config("configs/base.yaml")
    image_size = config["data"]["image_size"]

    train_images_dir = Path(config["dataset"]["splits"]["train"]["images_dir"])
    train_masks_dir = Path(config["dataset"]["splits"]["train"]["masks_dir"])
    val_images_dir = Path(config["dataset"]["splits"]["val"]["images_dir"])
    val_masks_dir = Path(config["dataset"]["splits"]["val"]["masks_dir"])

    for directory in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
        ensure_directory(directory)

    train_count = 4
    val_count = 2

    for i in range(train_count):
        image, mask = create_sample_image_and_mask(image_size=image_size, sample_index=i)
        save_sample(
            image=image,
            mask=mask,
            images_dir=train_images_dir,
            masks_dir=train_masks_dir,
            name=f"train_sample_{i:03d}",
        )

    for i in range(val_count):
        image, mask = create_sample_image_and_mask(image_size=image_size, sample_index=100 + i)
        save_sample(
            image=image,
            mask=mask,
            images_dir=val_images_dir,
            masks_dir=val_masks_dir,
            name=f"val_sample_{i:03d}",
        )

    print("=" * 60)
    print("Synthetic dataset generation complete")
    print("=" * 60)
    print(f"Train samples created: {train_count}")
    print(f"Validation samples created: {val_count}")
    print(f"Image size: {image_size}x{image_size}")
    print("\nSamples were written to:")
    print(f"- {train_images_dir}")
    print(f"- {train_masks_dir}")
    print(f"- {val_images_dir}")
    print(f"- {val_masks_dir}")


if __name__ == "__main__":
    main()