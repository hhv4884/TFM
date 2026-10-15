import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_degree_counts(ax, rows, degree_type):
    if rows.empty:
        ax.set_xlabel(f"{degree_type}-degree")
        ax.set_ylabel("Number of neurons")
        return
    degrees = rows["degree"].to_numpy(dtype=int)
    counts = rows["count"].to_numpy(dtype=int)
    bins = np.linspace(degrees.min(), degrees.max(), 36)
    ax.hist(
        degrees,
        bins=bins,
        weights=counts,
        edgecolor="black",
    )
    total_neurons = counts.sum()
    mean_degree = np.average(degrees, weights=counts)
    cumulative_counts = np.cumsum(counts)
    middle_positions = [
        (total_neurons - 1) // 2,
        total_neurons // 2,
    ]
    median_values = [
        degrees[np.searchsorted(cumulative_counts, pos + 1)] for pos in middle_positions
    ]
    median_degree = sum(median_values) / 2

    std_degree = np.sqrt(np.average((degrees - mean_degree) ** 2, weights=counts))
    ax.text(
        0.98,
        0.97,
        f"Mean: {mean_degree:.2f}\n"
        f"Median: {median_degree:.2f}\n"
        f"Std. dev.: {std_degree:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )
    ax.set_xlabel(f"{degree_type}-degree")
    ax.set_ylabel("Number of neurons")
    ax.grid(axis="y", alpha=0.3)


def plot_clustering_counts(ax, rows, neuron_type):
    ax.set_xlabel("Mean neighborhood Jaccard similarity")
    ax.set_ylabel("Number of neurons")
    if rows.empty:
        return
    values = rows["clustering_coefficient"].to_numpy(dtype=float)
    ax.hist(values, bins=np.linspace(0, 1, 36), edgecolor="black")
    ax.text(
        0.98,
        0.97,
        f"Mean: {np.mean(values):.4f}\n"
        f"Median: {np.median(values):.4f}\n"
        f"Std. dev.: {np.std(values):.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )
    ax.grid(axis="y", alpha=0.3)


def save_partition_svg_plots(partition_path, partition_number, epochs):
    degree_csv = os.path.join(partition_path, "degree_distribution.csv")
    clustering_csv = os.path.join(partition_path, "clustering_coefficient.csv")
    if os.path.exists(degree_csv):
        degree_df = pd.read_csv(degree_csv)
        for degree_type in ("in", "out"):
            fig, axes = plt.subplots(1, len(epochs), figsize=(15, 4), squeeze=False)
            axes = axes[0]
            for ax, epoch in zip(axes, epochs):
                rows = degree_df[
                    (degree_df["epoch"] == epoch)
                    & (degree_df["degree_type"] == degree_type)
                ][["degree", "count"]]
                plot_degree_counts(ax, rows, degree_type.capitalize())
                ax.set_title(f"Epoch {epoch}")
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    partition_path,
                    f"partition_{partition_number}_"
                    f"{degree_type}_degree_distribution.svg",
                ),
                format="svg",
                bbox_inches="tight",
            )
            plt.close(fig)
    if os.path.exists(clustering_csv):
        clustering_df = pd.read_csv(clustering_csv)
        for neuron_type in ("in", "out"):
            fig, axes = plt.subplots(1, len(epochs), figsize=(15, 4), squeeze=False)
            axes = axes[0]
            for ax, epoch in zip(axes, epochs):
                rows = clustering_df[
                    (clustering_df["epoch"] == epoch)
                    & (clustering_df["neuron_type"] == neuron_type)
                ]
                plot_clustering_counts(ax, rows, neuron_type)
                ax.set_title(f"Epoch {epoch}")

            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    partition_path,
                    f"partition_{partition_number}_" f"{neuron_type}_clustering.svg",
                ),
                format="svg",
                bbox_inches="tight",
            )
            plt.close(fig)
