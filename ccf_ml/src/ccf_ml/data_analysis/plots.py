"""Compactness pictures: 5-NN histogram and quantile-vs-isolation scatter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def plot_compactness_1d(
    *,
    q: np.ndarray,
    nn5: np.ndarray,
    title: str,
    destination: Path,
) -> None:
    """Histogram of 5-NN |d| and scatter of cell quantile versus 5-NN |d|."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axes[0].hist(nn5, bins=min(40, max(10, int(nn5.size // 2))), color="#334155", edgecolor="white")
    axes[0].set_xlabel("mean 5-NN |d|")
    axes[0].set_ylabel("frauds")
    axes[0].set_title("compactness")
    axes[1].scatter(q, nn5, s=12, alpha=0.7, c="#1d4ed8", linewidths=0)
    axes[1].set_xlabel("cell quantile q")
    axes[1].set_ylabel("mean 5-NN |d|")
    axes[1].set_title("tail vs isolated")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def plot_compactness_joint(
    *,
    nn5: np.ndarray,
    far5: np.ndarray,
    title: str,
    destination: Path,
) -> None:
    """Histogram of joint 5-NN distance and scatter against the 5-far mean."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axes[0].hist(nn5, bins=min(40, max(10, int(nn5.size // 2))), color="#334155", edgecolor="white")
    axes[0].set_xlabel("mean 5-NN joint |d|")
    axes[0].set_ylabel("frauds")
    axes[0].set_title("joint compactness")
    axes[1].scatter(far5, nn5, s=12, alpha=0.7, c="#b45309", linewidths=0)
    axes[1].set_xlabel("mean 5-far joint |d|")
    axes[1].set_ylabel("mean 5-NN joint |d|")
    axes[1].set_title("near vs far crowd")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)
