from typing import Any

from torch.utils.data import DataLoader

from src.data.dataset import build_dataset_from_config


def build_dataloader_from_config(
    config: dict[str, Any],
    split: str,
    shuffle: bool = False,
) -> DataLoader:
    dataset = build_dataset_from_config(config, split)

    num_workers = int(config["training"].get("num_workers", 0))
    pin_memory = bool(config["training"].get("pin_memory", False))

    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )