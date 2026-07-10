"""Training curve and embedding visualisation utilities.

Provides ``plot_training_curves`` for loss/metric history and
``plot_embeddings`` for 2-D projection scatter plots.

Expected config params: None — all functions accept explicit arguments.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

import numpy as np


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[str] = None,
) -> None:
    """Plot loss and metric curves over epochs.

    Args:
        history (Dict[str, List[float]]): Mapping of metric names to
            per-epoch values, e.g.
            ``{"train_loss": [...], "val_loss": [...]}``.
        title (str): Plot title.
        save_path (Optional[str]): If given, save the figure to this
            path instead of showing it interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_training_curves(). "
            "Install it with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, values in history.items():
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, marker="o", markersize=4, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "tsne",
    title: str = "Embedding Visualization",
    save_path: Optional[str] = None,
) -> None:
    """Project high-dimensional embeddings to 2-D and plot.

    Args:
        embeddings (np.ndarray): Array of shape ``(n_samples, dim)``.
        labels (Optional[np.ndarray]): Integer or string labels for
            colouring, shape ``(n_samples,)``.
        method (str): Projection method — ``"tsne"`` or ``"pca"``.
        title (str): Plot title.
        save_path (Optional[str]): If given, save the figure to this
            path instead of showing it interactively.

    Raises:
        ValueError: If *method* is not ``"tsne"`` or ``"pca"``.
        ImportError: If ``matplotlib`` or ``scikit-learn`` is missing.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_embeddings(). "
            "Install it with: pip install matplotlib"
        ) from exc

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for plot_embeddings(). "
            "Install it with: pip install scikit-learn"
        ) from exc

    method = method.lower()
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(
            f"Unknown projection method '{method}'. "
            f"Choose 'tsne' or 'pca'."
        )

    projected = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))

    if labels is not None:
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            mask = labels == lbl
            ax.scatter(
                projected[mask, 0],
                projected[mask, 1],
                label=str(lbl),
                alpha=0.7,
                s=20,
            )
        ax.legend(markerscale=2, fontsize=8)
    else:
        ax.scatter(
            projected[:, 0],
            projected[:, 1],
            alpha=0.7,
            s=20,
        )

    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
