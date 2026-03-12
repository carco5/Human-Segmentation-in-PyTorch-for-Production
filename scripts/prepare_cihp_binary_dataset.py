from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.config import load_config


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_ids(id_file: Path) -> list[str]:
    with id_file.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def find_image_path(images_dir: Path, sample_id: str, extensions: list[str]) -> Path:
    for extension in extensions:
        candidate = images_dir / f"{sample_id}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Image for sample '{sample_id}' not found in {images_dir} "
        f"with extensions {extensions}"
    )


def build_binary_mask(category_mask_path: Path) -> Image.Image:
    mask = Image.open(category_mask_path).convert("L")
    mask_array = np.asarray(mask, dtype=np.uint8)

    # CIHP category masks: background=0, human-part classes > 0
    binary_mask = (mask_array > 0).astype(np.uint8) * 255

    return Image.fromarray(binary_mask)


def process_split(
    sample_ids: list[str],
    raw_images_dir: Path,
    raw_category_masks_dir: Path,
    processed_images_dir: Path,
    processed_masks_dir: Path,
    image_extensions: list[str],
) -> None:
    ensure_directory(processed_images_dir)
    ensure_directory(processed_masks_dir)

    for sample_id in tqdm(sample_ids, desc=f"Processing {processed_images_dir.parent.name}"):
        image_path = find_image_path(raw_images_dir, sample_id, image_extensions)
        category_mask_path = raw_category_masks_dir / f"{sample_id}.png"

        if not category_mask_path.exists():
            raise FileNotFoundError(
                f"Category mask not found for sample '{sample_id}': {category_mask_path}"
            )

        output_image_path = processed_images_dir / f"{sample_id}.png"
        output_mask_path = processed_masks_dir / f"{sample_id}.png"

        image = Image.open(image_path).convert("RGB")
        image.save(output_image_path)

        binary_mask = build_binary_mask(category_mask_path)
        binary_mask.save(output_mask_path)


def main() -> None:
    config = load_config("configs/base.yaml")

    train_raw = config["dataset"]["raw_splits"]["train"]
    val_raw = config["dataset"]["raw_splits"]["val"]
    train_processed = config["dataset"]["splits"]["train"]
    val_processed = config["dataset"]["splits"]["val"]

    train_ids = read_ids(Path(train_raw["id_file"]))
    val_ids = read_ids(Path(val_raw["id_file"]))

    process_split(
        sample_ids=train_ids,
        raw_images_dir=Path(train_raw["images_dir"]),
        raw_category_masks_dir=Path(train_raw["category_masks_dir"]),
        processed_images_dir=Path(train_processed["images_dir"]),
        processed_masks_dir=Path(train_processed["masks_dir"]),
        image_extensions=config["dataset"]["image_extensions"],
    )

    process_split(
        sample_ids=val_ids,
        raw_images_dir=Path(val_raw["images_dir"]),
        raw_category_masks_dir=Path(val_raw["category_masks_dir"]),
        processed_images_dir=Path(val_processed["images_dir"]),
        processed_masks_dir=Path(val_processed["masks_dir"]),
        image_extensions=config["dataset"]["image_extensions"],
    )

    print("=" * 60)
    print("CIHP binary dataset preparation complete")
    print("=" * 60)
    print(f"Train samples processed: {len(train_ids)}")
    print(f"Validation samples processed: {len(val_ids)}")
    print(f"Processed dataset root: {config['dataset']['processed_root_dir']}")


if __name__ == "__main__":
    main()