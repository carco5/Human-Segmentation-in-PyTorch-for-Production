from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.config import load_config


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_rgb_image(image_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    x_gradient = np.linspace(40, 180, image_size, dtype=np.uint8)
    y_gradient = np.linspace(30, 150, image_size, dtype=np.uint8)

    image[..., 0] = x_gradient[np.newaxis, :]
    image[..., 1] = y_gradient[:, np.newaxis]
    image[..., 2] = rng.integers(70, 120, size=(image_size, image_size), dtype=np.uint8)

    return image


def build_category_mask(image_size: int, variant: int) -> np.ndarray:
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    yy, xx = np.ogrid[:image_size, :image_size]

    # Simulate multiple human-part labels (>0)
    if variant == 0:
        mask[50:210, 95:155] = 5      # torso
        mask[35:70, 105:145] = 2      # head
        mask[90:180, 70:95] = 8       # left arm
        mask[90:180, 155:180] = 10    # right arm
    elif variant == 1:
        center_x, center_y = 128, 130
        body = ((xx - center_x) ** 2) / (38 ** 2) + ((yy - center_y) ** 2) / (78 ** 2) <= 1
        head = ((xx - center_x) ** 2) / (24 ** 2) + ((yy - 60) ** 2) / (22 ** 2) <= 1
        mask[body] = 7
        mask[head] = 3
    else:
        mask[60:220, 110:150] = 12    # central body
        mask[50:85, 105:155] = 4       # head
        mask[150:230, 80:110] = 14     # left leg
        mask[150:230, 150:180] = 16    # right leg

    return mask


def tint_foreground(image: np.ndarray, category_mask: np.ndarray) -> np.ndarray:
    result = image.copy()
    foreground = category_mask > 0
    result[foreground, 0] = 220
    result[foreground, 1] = 90
    result[foreground, 2] = 90
    return result


def save_sample(images_dir: Path, masks_dir: Path, sample_id: str, image: np.ndarray, mask: np.ndarray) -> None:
    Image.fromarray(image).save(images_dir / f"{sample_id}.jpg")
    Image.fromarray(mask).save(masks_dir / f"{sample_id}.png")


def write_ids(id_file: Path, sample_ids: list[str]) -> None:
    id_file.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")


def main() -> None:
    config = load_config("configs/base.yaml")
    image_size = config["data"]["image_size"]

    train_images_dir = Path(config["dataset"]["raw_splits"]["train"]["images_dir"])
    train_masks_dir = Path(config["dataset"]["raw_splits"]["train"]["category_masks_dir"])
    train_id_file = Path(config["dataset"]["raw_splits"]["train"]["id_file"])

    val_images_dir = Path(config["dataset"]["raw_splits"]["val"]["images_dir"])
    val_masks_dir = Path(config["dataset"]["raw_splits"]["val"]["category_masks_dir"])
    val_id_file = Path(config["dataset"]["raw_splits"]["val"]["id_file"])

    for directory in [train_images_dir, train_masks_dir, val_images_dir, val_masks_dir]:
        ensure_directory(directory)

    train_ids = ["mock_train_000", "mock_train_001"]
    val_ids = ["mock_val_000"]

    for idx, sample_id in enumerate(train_ids):
        category_mask = build_category_mask(image_size=image_size, variant=idx)
        image = build_rgb_image(image_size=image_size, seed=idx)
        image = tint_foreground(image, category_mask)
        save_sample(train_images_dir, train_masks_dir, sample_id, image, category_mask)

    for idx, sample_id in enumerate(val_ids):
        category_mask = build_category_mask(image_size=image_size, variant=2 + idx)
        image = build_rgb_image(image_size=image_size, seed=100 + idx)
        image = tint_foreground(image, category_mask)
        save_sample(val_images_dir, val_masks_dir, sample_id, image, category_mask)

    write_ids(train_id_file, train_ids)
    write_ids(val_id_file, val_ids)

    print("=" * 60)
    print("Mock CIHP raw dataset created")
    print("=" * 60)
    print(f"Train IDs: {train_ids}")
    print(f"Val IDs:   {val_ids}")
    print(f"Image size: {image_size}x{image_size}")
    print("\nRaw structure created under:")
    print(f"- {Path(config['dataset']['raw_root_dir'])}")


if __name__ == "__main__":
    main()