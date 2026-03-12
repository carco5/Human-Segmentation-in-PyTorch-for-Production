import torch

from src.data.dataloader import build_dataloader_from_config
from src.utils.config import load_config


def describe_batch(name: str, batch: dict) -> None:
    images = batch["image"]
    masks = batch["mask"]

    print(f"{name} batch")
    print(f"- image tensor shape: {tuple(images.shape)}")
    print(f"- mask tensor shape:  {tuple(masks.shape)}")
    print(f"- image dtype: {images.dtype}")
    print(f"- mask dtype:  {masks.dtype}")
    print(f"- image min/max: {images.min().item():.4f} / {images.max().item():.4f}")
    print(f"- mask unique values: {torch.unique(masks).tolist()}")
    print()


def main() -> None:
    config = load_config("configs/base.yaml")

    train_loader = build_dataloader_from_config(config, split="train", shuffle=True)
    val_loader = build_dataloader_from_config(config, split="val", shuffle=False)

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print("=" * 60)
    print("DataLoader Smoke Test")
    print("=" * 60)
    describe_batch("Train", train_batch)
    describe_batch("Validation", val_batch)

    print("Status: data pipeline is working correctly.")


if __name__ == "__main__":
    main()