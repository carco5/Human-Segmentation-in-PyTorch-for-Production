from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import (
    dice_score_from_logits,
    foreground_ratio,
    foreground_ratio_from_logits,
    iou_score_from_logits,
)


def _resolve_total_batches(dataloader: DataLoader, max_batches: int | None) -> int:
    if max_batches is None:
        return len(dataloader)
    return min(len(dataloader), max_batches)


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    epsilon: float,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
    stage: str = "train",
) -> dict[str, float]:
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_pred_fg_ratio = 0.0
    total_target_fg_ratio = 0.0
    processed_batches = 0

    total_batches = _resolve_total_batches(dataloader, max_batches)

    progress_bar = tqdm(
        enumerate(dataloader),
        total=total_batches,
        desc=f"{stage.capitalize()} epoch",
    )

    for batch_idx, batch in progress_bar:
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, masks)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        batch_loss = loss.item()
        batch_dice = dice_score_from_logits(
            logits.detach(),
            masks.detach(),
            threshold=threshold,
            epsilon=epsilon,
        )
        batch_iou = iou_score_from_logits(
            logits.detach(),
            masks.detach(),
            threshold=threshold,
            epsilon=epsilon,
        )
        batch_pred_fg_ratio = foreground_ratio_from_logits(
            logits.detach(),
            threshold=threshold,
        )
        batch_target_fg_ratio = foreground_ratio(masks.detach())

        total_loss += batch_loss
        total_dice += batch_dice
        total_iou += batch_iou
        total_pred_fg_ratio += batch_pred_fg_ratio
        total_target_fg_ratio += batch_target_fg_ratio
        processed_batches += 1

        progress_bar.set_postfix(
            loss=f"{batch_loss:.4f}",
            dice=f"{batch_dice:.4f}",
            iou=f"{batch_iou:.4f}",
        )

    if processed_batches == 0:
        raise ValueError(f"No batches were processed during {stage} epoch.")

    return {
        "loss": total_loss / processed_batches,
        "dice": total_dice / processed_batches,
        "iou": total_iou / processed_batches,
        "pred_fg_ratio": total_pred_fg_ratio / processed_batches,
        "target_fg_ratio": total_target_fg_ratio / processed_batches,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float,
    epsilon: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    return run_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        threshold=threshold,
        epsilon=epsilon,
        max_batches=max_batches,
        stage="train",
    )


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    epsilon: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    return run_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=None,
        device=device,
        threshold=threshold,
        epsilon=epsilon,
        max_batches=max_batches,
        stage="val",
    )