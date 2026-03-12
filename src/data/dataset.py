from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BinarySegmentationDataset(Dataset):
    """
    Dataset for binary human segmentation.

    Expected structure:
        split/
            images/
                sample_001.jpg
            masks/
                sample_001.png

    The image and mask must share the same file stem.
    """

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        image_size: int,
        image_extensions: list[str] | None = None,
        mask_extension: str = ".png",
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.image_extensions = image_extensions or [".jpg", ".jpeg", ".png"]
        self.mask_extension = mask_extension

        self._validate_directories()
        self.samples = self._build_samples()

    def _validate_directories(self) -> None:
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

    def _build_samples(self) -> list[tuple[Path, Path]]:
        image_paths: list[Path] = []

        for extension in self.image_extensions:
            image_paths.extend(sorted(self.images_dir.glob(f"*{extension}")))

        if not image_paths:
            raise ValueError(f"No image files found in: {self.images_dir}")

        samples: list[tuple[Path, Path]] = []

        for image_path in image_paths:
            mask_path = self.masks_dir / f"{image_path.stem}{self.mask_extension}"

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Missing mask for image '{image_path.name}'. "
                    f"Expected mask path: {mask_path}"
                )

            samples.append((image_path, mask_path))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.uint8)

        # Enforce binary masks: any non-zero pixel becomes foreground
        mask_array = (mask_array > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def get_split_directories(config: dict[str, Any], split: str) -> tuple[Path, Path]:
    split_config = config["dataset"]["splits"][split]
    images_dir = Path(split_config["images_dir"])
    masks_dir = Path(split_config["masks_dir"])
    return images_dir, masks_dir


def build_dataset_from_config(
    config: dict[str, Any], split: str
) -> BinarySegmentationDataset:
    images_dir, masks_dir = get_split_directories(config, split)

    return BinarySegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        image_size=config["data"]["image_size"],
        image_extensions=config["dataset"]["image_extensions"],
        mask_extension=config["dataset"]["mask_extension"],
    )