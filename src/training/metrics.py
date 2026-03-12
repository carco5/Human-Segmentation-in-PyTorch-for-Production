import torch


def logits_to_binary_predictions(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= threshold).float()
    return predictions


def _flatten_tensors(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = predictions.reshape(predictions.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    return predictions, targets


def dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-6,
) -> float:
    epsilon = float(epsilon)
    predictions, targets = _flatten_tensors(predictions, targets)

    intersection = (predictions * targets).sum(dim=1)
    denominator = predictions.sum(dim=1) + targets.sum(dim=1)

    score = (2 * intersection + epsilon) / (denominator + epsilon)
    return score.mean().item()


def iou_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-6,
) -> float:
    epsilon = float(epsilon)
    predictions, targets = _flatten_tensors(predictions, targets)

    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection

    score = (intersection + epsilon) / (union + epsilon)
    return score.mean().item()


def dice_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    predictions = logits_to_binary_predictions(logits, threshold=threshold)
    return dice_score(predictions, targets, epsilon=epsilon)


def iou_score_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> float:
    predictions = logits_to_binary_predictions(logits, threshold=threshold)
    return iou_score(predictions, targets, epsilon=epsilon)

def foreground_ratio(mask: torch.Tensor) -> float:
    return mask.float().mean().item()


def foreground_ratio_from_logits(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    predictions = logits_to_binary_predictions(logits, threshold=threshold)
    return foreground_ratio(predictions)