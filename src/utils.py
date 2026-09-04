"""Small shared helpers: reproducibility, filesystem and plotting utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and TensorFlow (if installed) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:  # TensorFlow is optional for the classical (KNN/SVM) workflow
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (including parents) if missing and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_accuracy_comparison(
    model_names: Iterable[str],
    accuracies: Iterable[float],
    save_path: str | Path,
    title: str = "Model Accuracy Comparison",
) -> Path:
    """Render and save a bar chart comparing model accuracies (percent)."""
    import matplotlib

    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    ensure_dir(save_path.parent)

    names = list(model_names)
    values = [float(a) for a in accuracies]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(names, values, color=["#e74c3c", "#8e44ad", "#f1c40f"])

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path
