from pathlib import Path
from typing import Any

import torch

from src.models.unet import build_unet_from_config
from src.training.metrics import logits_to_binary_predictions


def get_checkpoint_path(config: dict[str, Any]) -> Path:
    experiment_name = config["experiment"]["name"]
    checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
    return checkpoints_dir / f"{experiment_name}_best.pt"


def load_model_from_checkpoint(
    config: dict[str, Any],
    device: torch.device,
    checkpoint_path: str | Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else get_checkpoint_path(config)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_unet_from_config(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


@torch.no_grad()
def predict_from_image_tensor(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    logits = model(image_tensor)
    probabilities = torch.sigmoid(logits)
    predictions = logits_to_binary_predictions(logits, threshold=threshold)

    return {
        "logits": logits,
        "probabilities": probabilities,
        "predictions": predictions,
    }