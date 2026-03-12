import torch

from src.training.losses import build_loss_from_config
from src.training.metrics import (
    dice_score_from_logits,
    iou_score_from_logits,
    logits_to_binary_predictions,
)
from src.utils.config import load_config


def main() -> None:
    config = load_config("configs/base.yaml")

    threshold = config["evaluation"]["threshold"]
    epsilon = float(config["evaluation"]["epsilon"])
    criterion = build_loss_from_config(config)

    targets = torch.tensor(
        [
            [
                [
                    [0, 0, 1, 1],
                    [0, 1, 1, 1],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0],
                ]
            ],
            [
                [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            ],
        ],
        dtype=torch.float32,
    )

    perfect_logits = torch.where(
        targets == 1.0,
        torch.full_like(targets, 8.0),
        torch.full_like(targets, -8.0),
    )

    wrong_logits = -perfect_logits

    perfect_loss = criterion(perfect_logits, targets).item()
    wrong_loss = criterion(wrong_logits, targets).item()

    perfect_dice = dice_score_from_logits(
        perfect_logits,
        targets,
        threshold=threshold,
        epsilon=epsilon,
    )
    perfect_iou = iou_score_from_logits(
        perfect_logits,
        targets,
        threshold=threshold,
        epsilon=epsilon,
    )

    wrong_dice = dice_score_from_logits(
        wrong_logits,
        targets,
        threshold=threshold,
        epsilon=epsilon,
    )
    wrong_iou = iou_score_from_logits(
        wrong_logits,
        targets,
        threshold=threshold,
        epsilon=epsilon,
    )

    perfect_predictions = logits_to_binary_predictions(perfect_logits, threshold=threshold)
    wrong_predictions = logits_to_binary_predictions(wrong_logits, threshold=threshold)

    print("=" * 60)
    print("Segmentation Metrics Smoke Test")
    print("=" * 60)

    print("Perfect prediction case")
    print(f"- loss: {perfect_loss:.6f}")
    print(f"- dice: {perfect_dice:.6f}")
    print(f"- iou:  {perfect_iou:.6f}")
    print(f"- unique predicted values: {torch.unique(perfect_predictions).tolist()}")
    print()

    print("Wrong prediction case")
    print(f"- loss: {wrong_loss:.6f}")
    print(f"- dice: {wrong_dice:.6f}")
    print(f"- iou:  {wrong_iou:.6f}")
    print(f"- unique predicted values: {torch.unique(wrong_predictions).tolist()}")
    print()

    assert perfect_loss < wrong_loss, "Perfect prediction should have lower loss than wrong prediction."
    assert abs(perfect_dice - 1.0) < 1e-5, "Perfect prediction should have Dice score ~ 1.0."
    assert abs(perfect_iou - 1.0) < 1e-5, "Perfect prediction should have IoU ~ 1.0."
    assert wrong_dice < 1e-4, "Wrong prediction should have Dice score close to 0."
    assert wrong_iou < 1e-4, "Wrong prediction should have IoU score close to 0."

    print("Status: loss function and segmentation metrics work correctly.")


if __name__ == "__main__":
    main()