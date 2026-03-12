from typing import Any

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = float(epsilon)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.reshape(probabilities.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
        dice_loss = 1.0 - dice_score.mean()

        return dice_loss


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(epsilon=epsilon)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def build_loss_from_config(config: dict[str, Any]) -> nn.Module:
    loss_config = config["loss"]
    loss_name = loss_config["name"]
    epsilon = float(config["evaluation"]["epsilon"])

    if loss_name == "bce_with_logits":
        return nn.BCEWithLogitsLoss()

    if loss_name == "bce_plus_dice":
        return BCEDiceLoss(
            bce_weight=float(loss_config["bce_weight"]),
            dice_weight=float(loss_config["dice_weight"]),
            epsilon=epsilon,
        )

    raise ValueError(f"Unsupported loss function: {loss_name}")