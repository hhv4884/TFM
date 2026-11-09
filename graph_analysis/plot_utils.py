import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_degree_counts(ax, rows, degree_type, degree_df, epoch, losses):
    if not rows.empty:
        degrees = rows["degree"].to_numpy(dtype=float)
        counts = rows["count"].to_numpy(dtype=int)
        min_degree = degrees.min()
        max_degree = degrees.max()
        if min_degree == max_degree:
            margin = max(abs(min_degree) * 0.05, 0.005)
            bins = np.linspace(min_degree - margin, max_degree + margin, 36)
        else:
            bins = np.linspace(min_degree, max_degree, 36)

        ax.hist(degrees, bins=bins, weights=counts, edgecolor="black")
    add_graph_info(ax, rows, degree_df, epoch, losses)
    ax.set_xlabel("Degree Index")
    ax.set_ylabel("Number of Neurons")
    ax.grid(axis="y", alpha=0.3)


def plot_clustering_counts(ax, rows, neuron_type, degree_df, epoch, losses):
    ax.set_xlabel("Clustering Index")
    ax.set_ylabel("Number of Neurons")
    if not rows.empty:
        values = rows["clustering_coefficient"].to_numpy(dtype=float)
        min_value = values.min()
        max_value = values.max()
        if min_value == max_value:
            margin = max(abs(min_value) * 0.05, 0.005)
            bins = np.linspace(min_value - margin, max_value + margin, 36)
        else:
            bins = np.linspace(min_value, max_value, 36)
        ax.hist(values, bins=bins, edgecolor="black")
    add_graph_info(ax, rows, degree_df, epoch, losses)
    ax.grid(axis="y", alpha=0.3)


def plot_strength_counts(ax, rows):
    if not rows.empty:
        values = rows["strength"].to_numpy(dtype=float)
        min_value = values.min()
        max_value = values.max()
        if min_value == max_value:
            margin = max(abs(min_value) * 0.05, 0.005)
            bins = np.linspace(min_value - margin, max_value + margin, 36)
        else:
            bins = np.linspace(min_value, max_value, 36)
        ax.hist(values, bins=bins, edgecolor="black")
        stats_text = (
            f"Mean: {np.mean(values):.4f}\n"
            f"Median: {np.median(values):.4f}\n"
            f"Std. dev.: {np.std(values):.4f}"
        )
        ax.text(
            0.98,
            0.97,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="gray",
                alpha=0.9,
            ),
        )
    ax.set_xlabel("Strength")
    ax.set_ylabel("Number of Neurons")
    ax.grid(axis="y", alpha=0.3)


def plot_weighted_clustering_counts(ax, rows):
    values = rows["clustering_coefficient"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) > 0:
        min_value = values.min()
        max_value = values.max()
        if min_value == max_value:
            margin = max(abs(min_value) * 0.05, 0.005)
            bins = np.linspace(
                min_value - margin,
                max_value + margin,
                36,
            )
        else:
            bins = np.linspace(min_value, max_value, 36)
        ax.hist(values, bins=bins, edgecolor="black")
        stats_text = (
            f"Mean: {np.mean(values):.4f}\n"
            f"Median: {np.median(values):.4f}\n"
            f"Std. dev.: {np.std(values):.4f}\n"
            f"Valid neurons: {len(values)}"
        )
        ax.text(
            0.98,
            0.97,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="gray",
                alpha=0.9,
            ),
        )
    ax.set_xlabel("Weighted signed bipartite clustering")
    ax.set_ylabel("Number of Neurons")
    ax.grid(axis="y", alpha=0.3)


def _normalize_degree_columns(degree_df):
    if degree_df.empty:
        return pd.DataFrame(columns=["epoch", "degree_type", "degree", "count"])
    degree_df = degree_df.copy()
    if "neuron_type" in degree_df.columns:
        degree_df = degree_df.rename(columns={"neuron_type": "degree_type"})
    required = {"epoch", "degree_type", "degree", "count"}
    missing = required - set(degree_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en degree_index.csv: {sorted(missing)}")
    degree_df["epoch"] = pd.to_numeric(degree_df["epoch"], errors="coerce")
    degree_df["degree"] = pd.to_numeric(degree_df["degree"], errors="coerce")
    degree_df["count"] = pd.to_numeric(degree_df["count"], errors="coerce")
    return degree_df.dropna(subset=["epoch", "degree", "count"])


def _normalize_clustering_columns(clustering_df):
    if clustering_df.empty:
        return pd.DataFrame(columns=["epoch", "neuron_type", "clustering_coefficient"])
    clustering_df = clustering_df.copy()
    required = {"epoch", "neuron_type", "clustering_coefficient"}
    missing = required - set(clustering_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en clustering_index.csv: {sorted(missing)}")
    clustering_df["epoch"] = pd.to_numeric(clustering_df["epoch"], errors="coerce")
    clustering_df["clustering_coefficient"] = pd.to_numeric(
        clustering_df["clustering_coefficient"], errors="coerce"
    )
    return clustering_df.dropna(subset=["epoch", "clustering_coefficient"])


def _normalize_strength_columns(strength_data):
    required_columns = [
        "epoch",
        "hidden_layer",
        "neuron_index",
        "input_strength",
        "output_strength",
        "strength",
    ]
    if strength_data is None or strength_data.empty:
        return pd.DataFrame(columns=required_columns)
    df = strength_data.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    )
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in strength data: {missing_columns}")
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["epoch", "hidden_layer", "neuron_index", "strength"])


def _normalize_weighted_clustering_columns(data):
    required_columns = [
        "epoch",
        "hidden_layer",
        "neuron_index",
        "clustering_coefficient",
    ]
    if data is None or data.empty:
        return pd.DataFrame(columns=required_columns)
    df = data.copy()
    df.columns = (
        df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=False)
    )
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in weighted clustering data: {missing}")
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["epoch", "hidden_layer", "neuron_index"])


def save_partition_pdf_plots(
    partition_path,
    partition_number,
    epochs,
    losses,
    degree_data=None,
    clustering_data=None,
    strength_data=None,
    weighted_clustering_data=None,
    hidden_layer=None,
):
    os.makedirs(partition_path, exist_ok=True)
    degree_df = _normalize_degree_columns(
        degree_data if degree_data is not None else pd.DataFrame()
    )
    clustering_df = _normalize_clustering_columns(
        clustering_data if clustering_data is not None else pd.DataFrame()
    )
    strength_df = _normalize_strength_columns(
        strength_data if strength_data is not None else pd.DataFrame()
    )
    weighted_clustering_df = _normalize_weighted_clustering_columns(
        weighted_clustering_data
        if weighted_clustering_data is not None
        else pd.DataFrame()
    )
    if not degree_df.empty:
        for degree_type in ("in", "out"):
            fig, axes = plt.subplots(1, len(epochs), figsize=(15, 4), squeeze=False)
            axes = axes[0]
            for ax, epoch in zip(axes, epochs):
                rows = degree_df[
                    (degree_df["epoch"] == epoch)
                    & (degree_df["degree_type"] == degree_type)
                ][["degree", "count"]]
                plot_degree_counts(
                    ax, rows, degree_type.capitalize(), degree_df, epoch, losses
                )
                ax.set_title(f"Epoch {epoch}")
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    partition_path,
                    f"partition_{partition_number}_"
                    f"{degree_type}_degree_distribution.pdf",
                ),
                format="pdf",
                bbox_inches="tight",
            )
            plt.close(fig)
    if not clustering_df.empty:
        for neuron_type in ("in", "out"):
            fig, axes = plt.subplots(1, len(epochs), figsize=(15, 4), squeeze=False)
            axes = axes[0]
            for ax, epoch in zip(axes, epochs):
                rows = clustering_df[
                    (clustering_df["epoch"] == epoch)
                    & (clustering_df["neuron_type"] == neuron_type)
                ]
                plot_clustering_counts(ax, rows, neuron_type, degree_df, epoch, losses)
                ax.set_title(f"Epoch {epoch}")
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    partition_path,
                    f"partition_{partition_number}_{neuron_type}_clustering.pdf",
                ),
                format="pdf",
                bbox_inches="tight",
            )
            plt.close(fig)

    if not strength_df.empty:
        strength_df = strength_df[strength_df["hidden_layer"] == hidden_layer]
        if not strength_df.empty:
            fig, axes = plt.subplots(1, len(epochs), figsize=(15, 4), squeeze=False)
            axes = axes[0]
            for ax, epoch in zip(axes, epochs):
                rows = strength_df[strength_df["epoch"] == epoch]
                plot_strength_counts(ax, rows)
                ax.set_title(f"Epoch {epoch}")
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    partition_path,
                    f"hidden_layer_{hidden_layer}_strength_distribution.pdf",
                ),
                format="pdf",
                bbox_inches="tight",
            )
            plt.close(fig)

    if not weighted_clustering_df.empty:
        weighted_clustering_df = weighted_clustering_df[
            weighted_clustering_df["hidden_layer"] == hidden_layer
        ]
        if not weighted_clustering_df.empty:
            fig, axes = plt.subplots(
                1,
                len(epochs),
                figsize=(15, 4),
                squeeze=False,
            )
            axes = axes[0]
            for ax, epoch in zip(axes, epochs):
                rows = weighted_clustering_df[weighted_clustering_df["epoch"] == epoch]
                plot_weighted_clustering_counts(ax, rows)
                ax.set_title(f"Epoch {epoch}")
            fig.tight_layout()
            fig.savefig(
                os.path.join(
                    partition_path,
                    f"hidden_layer_{hidden_layer}_weighted_clustering_distribution.pdf",
                ),
                format="pdf",
                bbox_inches="tight",
            )
            plt.close(fig)


def load_losses_from_csv(epochs_info_path):
    df = pd.read_csv(epochs_info_path)
    required = {"epoch", "train_loss", "test_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en epochs_info.csv: {sorted(missing)}")
    for column in ("epoch", "train_loss", "test_loss"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["epoch", "train_loss", "test_loss"])
    if df.empty:
        raise ValueError("epochs_info.csv no contiene épocas válidas.")
    df = df.drop_duplicates(subset=["epoch"], keep="last")
    df["epoch"] = df["epoch"].astype(int)
    losses = {
        int(row.epoch): {
            "train_loss": float(row.train_loss),
            "test_loss": float(row.test_loss),
        }
        for row in df.itertuples(index=False)
    }
    best_row = df.loc[df["test_loss"].idxmin()]
    best_epoch = int(best_row["epoch"])
    return losses, best_epoch


def load_losses_from_logs(logs_path):
    losses = {}
    pattern = re.compile(
        r"Epoch \[(\d+)/\d+\],\s*"
        r"Step \[(\d+)/\d+\].*?"
        r"Train Loss:\s*([0-9.eE+-]+).*?"
        r"Test Loss:\s*([0-9.eE+-]+)"
    )
    if not os.path.isdir(logs_path):
        return losses
    for filename in os.listdir(logs_path):
        if not filename.startswith("log_epoch_") or not filename.endswith(".txt"):
            continue
        filepath = os.path.join(logs_path, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epoch = int(match.group(1)) - 1
                    losses[epoch] = {
                        "train_loss": float(match.group(3)),
                        "test_loss": float(match.group(4)),
                    }
    return losses


def add_graph_info(ax, rows, degree_df, epoch, losses):
    epoch_rows = degree_df[degree_df["epoch"] == epoch]
    in_rows = epoch_rows[epoch_rows["degree_type"] == "in"]
    out_rows = epoch_rows[epoch_rows["degree_type"] == "out"]
    n_nodes = int(in_rows["count"].sum() + out_rows["count"].sum())
    n_edges = int((in_rows["degree"] * in_rows["count"]).sum())
    epoch_losses = losses.get(epoch)
    if epoch_losses is None:
        train_loss_text = "N/A"
        test_loss_text = "N/A"
    else:
        train_loss_text = f"{epoch_losses['train_loss']:.4f}"
        test_loss_text = f"{epoch_losses['test_loss']:.4f}"
    if not rows.empty and "degree" in rows.columns:
        degrees = rows["degree"].to_numpy(dtype=float)
        counts = rows["count"].to_numpy(dtype=float)
        mean = np.average(degrees, weights=counts)
        cumulative_counts = np.cumsum(counts)
        total = int(counts.sum())
        if total > 0:
            middle_positions = [(total - 1) // 2, total // 2]
            median_values = [
                degrees[np.searchsorted(cumulative_counts, pos + 1)]
                for pos in middle_positions
            ]
            median = sum(median_values) / 2
            std = np.sqrt(np.average((degrees - mean) ** 2, weights=counts))
            stats_text = (
                f"Mean: {mean:.2f}\n"
                f"Median: {median:.2f}\n"
                f"Std. dev.: {std:.2f}\n"
            )
        else:
            stats_text = ""
    elif not rows.empty and "clustering_coefficient" in rows.columns:
        values = rows["clustering_coefficient"].to_numpy(dtype=float)
        stats_text = (
            f"Mean: {np.mean(values):.4f}\n"
            f"Median: {np.median(values):.4f}\n"
            f"Std. dev.: {np.std(values):.4f}\n"
        )
    else:
        stats_text = ""
    ax.text(
        0.98,
        0.97,
        stats_text
        + f"Nodes: {n_nodes}\n"
        + f"Edges: {n_edges}\n"
        + f"Train Loss: {train_loss_text}\n"
        + f"Test Loss: {test_loss_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )


def save_training_metrics_plot(
    epoch_info_path,
    degree_metrics_path,
    clustering_metrics_path,
    output_path,
    best_epoch,
):
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    losses_df = pd.read_csv(epoch_info_path)
    degree_df = pd.read_csv(degree_metrics_path)
    clustering_df = pd.read_csv(clustering_metrics_path)
    required_losses = {"epoch", "train_loss", "test_loss"}
    required_metrics = {"epoch", "partition", "neuron_type", "mean", "std_dev"}
    if not required_losses.issubset(losses_df.columns):
        raise ValueError(
            f"Faltan columnas en {epoch_info_path}: "
            f"{required_losses - set(losses_df.columns)}"
        )
    for name, df in [
        ("degree_index_metrics.csv", degree_df),
        ("clustering_index_metrics.csv", clustering_df),
    ]:
        missing = required_metrics - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en {name}: {missing}")
    for col in ("epoch", "train_loss", "test_loss"):
        losses_df[col] = pd.to_numeric(losses_df[col], errors="coerce")
    losses_df = losses_df.dropna(subset=["epoch", "train_loss", "test_loss"])
    losses_df["epoch"] = losses_df["epoch"].astype(int)
    losses_df = losses_df.drop_duplicates(subset=["epoch"], keep="last").sort_values(
        "epoch"
    )
    for df in (degree_df, clustering_df):
        for col in ("epoch", "partition", "mean", "std_dev"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(
            subset=["epoch", "partition", "mean", "std_dev"],
            inplace=True,
        )
        df["epoch"] = df["epoch"].astype(int)
        df["partition"] = df["partition"].astype(int)
    partitions = sorted(
        set(degree_df["partition"].unique()) | set(clustering_df["partition"].unique())
    )
    for partition in partitions:
        for neuron_type in ("in", "out"):
            degree_subset = degree_df[
                (degree_df["partition"] == partition)
                & (degree_df["neuron_type"] == neuron_type)
            ].sort_values("epoch")
            clustering_subset = clustering_df[
                (clustering_df["partition"] == partition)
                & (clustering_df["neuron_type"] == neuron_type)
            ].sort_values("epoch")
            if degree_subset.empty or clustering_subset.empty:
                print(
                    f"Aviso: faltan datos para la partición "
                    f"{partition}, tipo {neuron_type}."
                )
                continue
            fig = plt.figure(figsize=(16, 4))
            gs = fig.add_gridspec(
                1,
                2,
                width_ratios=[3.5, 1.2],
                wspace=0.25,
            )
            ax1 = fig.add_subplot(gs[0, 0])
            ax_loss = fig.add_subplot(gs[0, 1])
            axes_y = [ax1]
            for _ in range(3):
                axes_y.append(ax1.twinx())
            ax1.yaxis.set_label_position("left")
            ax1.yaxis.tick_left()
            ax1.spines["left"].set_position(("outward", 0))
            for i, ax in enumerate(axes_y[1:], start=1):
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_visible(True)
                ax.yaxis.set_label_position("left")
                ax.yaxis.tick_left()
                ax.spines["left"].set_position(("outward", 55 * i))
                ax.patch.set_visible(False)
            curves = [
                (
                    losses_df["epoch"],
                    losses_df["test_loss"],
                    "Test loss",
                    "tab:orange",
                    "-",
                ),
                (
                    clustering_subset["epoch"],
                    clustering_subset["mean"],
                    "Media clustering",
                    "tab:green",
                    "-",
                ),
                (
                    clustering_subset["epoch"],
                    clustering_subset["std_dev"],
                    "Desv. clustering",
                    "tab:red",
                    "-",
                ),
                (
                    degree_subset["epoch"],
                    degree_subset["std_dev"],
                    "Desv. grado",
                    "tab:brown",
                    "-",
                ),
            ]
            lines = []
            for ax, curve in zip(axes_y, curves):
                x, y, label, color, linestyle = curve
                (line,) = ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=label,
                )
                ax.set_ylabel(label, color=color)
                ax.tick_params(axis="y", colors=color)
                ax.spines["left"].set_color(color)
                lines.append(line)
            ax1.axvline(
                x=best_epoch,
                color="black",
                linestyle=":",
                linewidth=2,
            )
            ax1.set_xlabel("Época")
            ax1.set_title(f"Partición {partition} — tipo {neuron_type}")
            ax1.grid(axis="x", alpha=0.25)
            (train_loss_line,) = ax_loss.plot(
                losses_df["train_loss"],
                losses_df["epoch"],
                color="tab:blue",
                linestyle="-",
                linewidth=1.7,
                label="Train loss",
            )
            ax_loss.axhline(
                y=best_epoch,
                color="black",
                linestyle=":",
                linewidth=2,
            )
            ax_loss.set_xlabel("Train loss")
            ax_loss.set_ylabel("Época")
            ax_loss.set_title("Train loss vs época")
            ax_loss.grid(alpha=0.25)
            lines.append(train_loss_line)
            best_epoch_line = plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle=":",
                label=f"Best Epoch ({best_epoch})",
            )
            lines.append(best_epoch_line)
            fig.legend(
                handles=lines,
                loc="center left",
                bbox_to_anchor=(0.85, 0.5),
                fontsize=9,
                frameon=False,
            )
            fig.subplots_adjust(
                left=0.30,
                right=0.80,
                bottom=0.18,
                top=0.85,
                wspace=0.25,
            )
            filename = f"training_metrics_partition_{partition}_" f"{neuron_type}.pdf"
            fig.savefig(
                os.path.join(output_dir, filename),
                format="pdf",
                bbox_inches="tight",
            )
            plt.close(fig)


def save_weighted_graph_training_metrics(
    epoch_info_path,
    strength_metrics_path,
    clustering_metrics_path,
    output_path,
    best_epoch,
):
    os.makedirs(output_path, exist_ok=True)
    losses_df = pd.read_csv(epoch_info_path)
    required_losses = {"epoch", "train_loss", "test_loss"}
    missing = required_losses - set(losses_df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en epochs_info.csv: {missing}")
    for col in ("epoch", "train_loss", "test_loss"):
        losses_df[col] = pd.to_numeric(losses_df[col], errors="coerce")
    losses_df = losses_df.dropna(subset=["epoch", "train_loss", "test_loss"])
    losses_df["epoch"] = losses_df["epoch"].astype(int)
    losses_df = losses_df.drop_duplicates(subset=["epoch"], keep="last").sort_values(
        "epoch"
    )
    strength_df = (
        pd.read_csv(strength_metrics_path)
        if strength_metrics_path is not None and os.path.isfile(strength_metrics_path)
        else pd.DataFrame()
    )
    clustering_df = (
        pd.read_csv(clustering_metrics_path)
        if clustering_metrics_path is not None
        and os.path.isfile(clustering_metrics_path)
        else pd.DataFrame()
    )
    for df in (strength_df, clustering_df):
        if not df.empty:
            df.columns = (
                df.columns.astype(str)
                .str.strip()
                .str.replace("\ufeff", "", regex=False)
            )
    for name, df in (
        ("strength_metrics", strength_df),
        ("weighted_clustering_metrics", clustering_df),
    ):
        if not df.empty:
            required = {"epoch", "hidden_layer", "mean", "std_dev"}
            missing = required - set(df.columns)
            if missing:
                raise ValueError(f"Faltan columnas en {name}: {missing}")
            for col in ("epoch", "hidden_layer", "mean", "std_dev"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df.dropna(subset=["epoch", "hidden_layer"], inplace=True)
            df["epoch"] = df["epoch"].astype(int)
            df["hidden_layer"] = df["hidden_layer"].astype(int)
    hidden_layers = set()
    if not strength_df.empty:
        hidden_layers.update(strength_df["hidden_layer"].unique())
    if not clustering_df.empty:
        hidden_layers.update(clustering_df["hidden_layer"].unique())
    for hidden_layer in sorted(hidden_layers):
        strength_subset = (
            strength_df[strength_df["hidden_layer"] == hidden_layer].sort_values(
                "epoch"
            )
            if not strength_df.empty
            else pd.DataFrame()
        )
        clustering_subset = (
            clustering_df[clustering_df["hidden_layer"] == hidden_layer].sort_values(
                "epoch"
            )
            if not clustering_df.empty
            else pd.DataFrame()
        )
        curves = []
        if not strength_subset.empty:
            curves.extend(
                [
                    (
                        strength_subset["epoch"],
                        strength_subset["mean"],
                        "Mean strength",
                        "tab:blue",
                    ),
                    (
                        strength_subset["epoch"],
                        strength_subset["std_dev"],
                        "Strength std. dev.",
                        "tab:cyan",
                    ),
                ]
            )
        if not clustering_subset.empty:
            curves.extend(
                [
                    (
                        clustering_subset["epoch"],
                        clustering_subset["mean"],
                        "Mean weighted clustering",
                        "tab:green",
                    ),
                    (
                        clustering_subset["epoch"],
                        clustering_subset["std_dev"],
                        "Clustering std. dev.",
                        "tab:red",
                    ),
                ]
            )
        curves.append(
            (
                losses_df["epoch"],
                losses_df["test_loss"],
                "Test loss",
                "tab:orange",
            )
        )
        fig = plt.figure(figsize=(16, 5))
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[3.5, 1.2],
            wspace=0.25,
        )
        ax1 = fig.add_subplot(gs[0, 0])
        ax_loss = fig.add_subplot(gs[0, 1])
        axes_y = [ax1]
        for _ in range(len(curves) - 1):
            axes_y.append(ax1.twinx())
        lines = []
        for i, (axis, curve) in enumerate(zip(axes_y, curves)):
            x, y, label, color = curve
            (line,) = axis.plot(
                x,
                y,
                color=color,
                linestyle="-",
                linewidth=1.7,
                label=label,
            )
            axis.set_ylabel(label, color=color)
            axis.tick_params(axis="y", colors=color)
            axis.yaxis.set_label_position("left")
            axis.yaxis.tick_left()
            axis.spines["right"].set_visible(False)
            axis.spines["left"].set_visible(True)
            axis.spines["left"].set_position(("outward", 50 * i))
            axis.spines["left"].set_color(color)
            if i > 0:
                axis.patch.set_visible(False)

            lines.append(line)
        ax1.axvline(
            x=best_epoch,
            color="black",
            linestyle=":",
            linewidth=2,
            label=f"Best epoch ({best_epoch})",
        )
        ax1.set_xlabel("Epoch")
        ax1.set_title(f"Weighted graph metrics — hidden layer {hidden_layer}")
        ax1.grid(axis="x", alpha=0.25)
        (train_loss_line,) = ax_loss.plot(
            losses_df["train_loss"],
            losses_df["epoch"],
            color="tab:blue",
            linestyle="-",
            linewidth=1.7,
            label="Train loss",
        )
        ax_loss.set_xlabel("Train loss")
        ax_loss.set_ylabel("Epoch")
        ax_loss.set_title("Train loss vs epoch")
        ax_loss.grid(alpha=0.25)
        lines.append(ax1.get_lines()[-1])
        lines.append(train_loss_line)
        fig.legend(
            handles=lines,
            loc="center left",
            bbox_to_anchor=(0.82, 0.5),
            fontsize=9,
            frameon=False
        )
        fig.subplots_adjust(
            left=0.30,
            right=0.80,
            bottom=0.18,
            top=0.85,
            wspace=0.25,
        )
        fig.savefig(
            os.path.join(
                output_path,
                f"training_metrics_hidden_layer_{hidden_layer}.pdf",
            ),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig)
