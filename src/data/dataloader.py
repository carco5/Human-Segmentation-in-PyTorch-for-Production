from typing import Any

from torch.utils.data import DataLoader

from src.data.dataset import build_dataset_from_config


def build_dataloader_from_config(
    config: dict[str, Any],
    split: str,
    shuffle: bool = False,
) -> DataLoader:
    dataset = build_dataset_from_config(config, split)

    return DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=shuffle,
        num_workers=0,
    )