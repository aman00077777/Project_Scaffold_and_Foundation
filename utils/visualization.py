"""
Visualization utilities for the Fusion framework.
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

def plot_embeddings_tsne(
    embeddings: np.ndarray,
    labels: List,
    output_path: str,
) -> None:
    """
    Visualize embeddings using t-SNE.

    Args:
        embeddings: Feature embeddings.
        labels: Labels corresponding to embeddings.
        output_path: Path to save the figure.
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))

    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels,
        cmap="tab10",
    )

    plt.colorbar(scatter)
    plt.title("t-SNE Embedding Visualization")

    plt.savefig(output_path)
    plt.close()

def plot_attention_map(
    attention_weights: torch.Tensor,
    tokens: List[str],
    output_path: str,
) -> None:
    """
    Plot an attention heatmap.

    Args:
        attention_weights: Attention weight matrix.
        tokens: Token labels.
        output_path: Path to save the figure.
    """
    plt.figure(figsize=(8, 6))

    plt.imshow(
        attention_weights.cpu().numpy(),
        cmap="viridis",
        aspect="auto",
    )

    plt.colorbar()

    plt.xticks(
        range(len(tokens)),
        tokens,
        rotation=90,
    )

    plt.yticks(
        range(len(tokens)),
        tokens,
    )

    plt.title("Attention Map")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_training_curves(
    metrics_history: Dict[str, List[float]],
    output_path: str,
) -> None:
    """
    Plot training and validation curves.

    Args:
        metrics_history: Dictionary containing training history.
        output_path: Path to save the figure.
    """
    plt.figure(figsize=(12, 5))

    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(metrics_history["train_loss"], label="Train Loss")
    plt.plot(metrics_history["val_loss"], label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(metrics_history["train_accuracy"], label="Train Accuracy")
    plt.plot(metrics_history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



