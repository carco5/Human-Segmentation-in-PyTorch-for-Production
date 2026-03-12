from typing import Any

import torch.nn as nn


def build_loss_from_config(config: dict[str, Any]) -> nn.Module:
    loss_name = config["loss"]["name"]

    if loss_name == "bce_with_logits":
        return nn.BCEWithLogitsLoss()

    raise ValueError(f"Unsupported loss function: {loss_name}")