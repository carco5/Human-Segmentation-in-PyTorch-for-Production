import torch
import torch.nn as nn

from src.data.dataloader import build_dataloader_from_config
from src.models.unet import build_unet_from_config
from src.utils.config import load_config


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def main() -> None:
    config = load_config("configs/base.yaml")

    device = torch.device(config["training"]["device"])

    model = build_unet_from_config(config).to(device)
    model.train()

    train_loader = build_dataloader_from_config(config, split="train", shuffle=False)
    batch = next(iter(train_loader))

    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    logits = model(images)
    loss = criterion(logits, masks)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    probabilities = torch.sigmoid(logits)

    print("=" * 60)
    print("U-Net Smoke Test")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Input shape:  {tuple(images.shape)}")
    print(f"Mask shape:   {tuple(masks.shape)}")
    print(f"Logit shape:  {tuple(logits.shape)}")
    print(f"Loss value:   {loss.item():.6f}")
    print(f"Prob min/max: {probabilities.min().item():.6f} / {probabilities.max().item():.6f}")

    shape_ok = logits.shape == masks.shape
    print(f"Shape compatibility: {shape_ok}")

    if not shape_ok:
        raise ValueError(
            f"Model output shape {tuple(logits.shape)} does not match mask shape {tuple(masks.shape)}"
        )

    print("\nStatus: U-Net forward, loss, backward, and optimizer step all work correctly.")


if __name__ == "__main__":
    main()