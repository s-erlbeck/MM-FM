#!/usr/bin/env python3
"""
Analyze the mixture-weight distribution of a fitted CLS-token GMM.

Loads the GaussianMixture stored under the "model" key of a GMM pickle
(see fit_gmm_imagenet.py), plots the distributions of soft assignment 
(mixture weights) and the hard assigment (arg-max component) and 
compares there Shannon entropies with the uniform entropy.

Usage:
    python src/scripts/analyze_gmm_weights.py \
        --gmm-path artifacts/gmm/dinov2-b/gmm_n8192_diag_k-means++.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy


def load_gmm_data(gmm_path: str):
    with open(gmm_path, "rb") as f:
        data = pickle.load(f)
    weights = data["model"].weights_
    labels = data["labels"]

    n_components = len(weights)
    n_data = len(labels)
    hard_fractions = np.bincount(labels, minlength=n_components) / n_data
    return weights, hard_fractions, n_data


def plot_weights(weights: np.ndarray, hard_fractions: np.ndarray, n_samples: int):
    n = len(weights)
    uniform_weight = 1.0 / n

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.boxplot([weights, hard_fractions],
               tick_labels=["Soft (weights)", "Hard (fractions)"])
    ax.axhline(uniform_weight, linestyle="--", linewidth=1,
               label=f"Uniform (1/{n})")
    ax.set_ylabel("Fraction of assignment mass")
    ax.set_title("Soft vs. hard assignment distribution")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    size_axis = ax.secondary_yaxis("right", functions=(
        lambda w: w * n_samples, lambda size: size / n_samples))
    size_axis.set_ylabel("Cluster size")

    ax = axes[1]
    ax.hist(weights, bins=50, edgecolor="white")
    ax.axvline(uniform_weight, linestyle="--", linewidth=1,
               label=f"Uniform (1/{n})")
    ax.set_xlabel("Mixture weight")
    ax.set_ylabel("Number of components")
    ax.set_title("Soft assignment: distribution of mixture weights")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    ax = axes[2]
    ax.hist(hard_fractions, bins=50, edgecolor="white")
    ax.axvline(uniform_weight, linestyle="--", linewidth=1,
               label=f"Uniform (1/{n})")
    ax.set_xlabel("Fraction of samples assigned")
    ax.set_ylabel("Number of components")
    ax.set_title("Hard assignment: distribution of cluster sizes")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the mixture-weight distribution of a fitted GMM"
    )
    parser.add_argument(
        "--gmm-path", type=str,
        default="artifacts/gmm/dinov2-b/gmm_n8192_diag_k-means++.pkl",
        help="Path to the GMM pickle (must contain a 'model' key)",
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Where to save the plot (default: next to this script, "
             "<name>_weights_analysis.png)",
    )
    args = parser.parse_args()

    weights, hard_fractions, n_samples = load_gmm_data(args.gmm_path)
    n = len(weights)

    uniform_entropy = entropy(np.full(n, 1.0 / n))
    mixture_entropy = entropy(weights)
    hard_entropy = entropy(hard_fractions)

    print(f"Loaded {n} mixture weights from {args.gmm_path}")
    print(f"Shannon entropy uniform (n={n}): {uniform_entropy:.4f} nats")
    print(f"Shannon entropy soft / mixture: {mixture_entropy:.4f} nats")
    print(f"Shannon entropy hard assignment: {hard_entropy:.4f} nats")
    print(f"Effective number of soft components: {np.exp(mixture_entropy):.1f}")
    print(f"Effective number of hard components: {np.exp(hard_entropy):.1f}")

    output_path = args.output_path
    if output_path is None:
        script_dir = Path(__file__).resolve().parent
        output_path = str(script_dir / f"{Path(args.gmm_path).stem}_weights_analysis.png")
    fig = plot_weights(weights, hard_fractions, n_samples)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
