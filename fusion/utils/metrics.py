"""
Evaluation metrics for the Fusion framework.
"""

import numpy as np
import torch
import torch.nn.functional as F

def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Compute top-1 classification accuracy.

    Args:
        logits: Model output logits.
        labels: Ground-truth labels.

    Returns:
        Accuracy as a float.
    """
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float().mean()

    return correct.item()

def compute_top_k_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int
) -> float:
    """
    Compute top-k classification accuracy.

    Args:
        logits: Model output logits.
        labels: Ground-truth labels.
        k: Number of top predictions to consider.

    Returns:
        Top-k accuracy as a float.
    """
    _, predictions = torch.topk(logits, k, dim=1)
    correct = predictions.eq(labels.view(-1, 1))
    accuracy = correct.any(dim=1).float().mean()

    return accuracy.item()

def compute_map(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute Mean Average Precision (mAP) for retrieval.

    Args:
        embeddings: Feature embeddings.
        labels: Ground-truth labels.

    Returns:
        Mean Average Precision.
    """
    embeddings = embeddings / np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )

    similarity = embeddings @ embeddings.T

    average_precisions = []

    for i in range(len(labels)):
        ranking = np.argsort(-similarity[i])

        ranking = ranking[ranking != i]

        relevant = (labels[ranking] == labels[i]).astype(int)

        if relevant.sum() == 0:
            continue

        precision = np.cumsum(relevant) / (
            np.arange(len(relevant)) + 1
        )

        average_precision = (
            precision * relevant
        ).sum() / relevant.sum()

        average_precisions.append(average_precision)

    if not average_precisions:
        return 0.0

    return float(np.mean(average_precisions))

def compute_ndcg(
    scores: np.ndarray,
    relevance: np.ndarray,
    k: int
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG).

    Args:
        scores: Predicted relevance scores.
        relevance: Ground-truth relevance scores.
        k: Number of top results to evaluate.

    Returns:
        NDCG score.
    """
    order = np.argsort(scores)[::-1][:k]
    ranked_relevance = relevance[order]

    discounts = np.log2(np.arange(2, len(ranked_relevance) + 2))
    dcg = np.sum(ranked_relevance / discounts)

    ideal_order = np.argsort(relevance)[::-1][:k]
    ideal_relevance = relevance[ideal_order]
    ideal_dcg = np.sum(
        ideal_relevance /
        np.log2(np.arange(2, len(ideal_relevance) + 2))
    )

    if ideal_dcg == 0:
        return 0.0

    return float(dcg / ideal_dcg)