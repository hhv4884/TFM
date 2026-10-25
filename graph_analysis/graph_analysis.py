import os
import pandas as pd
from .plot_utils import (
    save_partition_pdf_plots,
    load_losses_from_csv,
    save_training_metrics_plot,
)


def create_separated_panels(model_path, graph_path, logs_path, epochs):
    epochs_info_path = os.path.join(graph_path, "epochs_info.csv")
    if not os.path.isfile(epochs_info_path):
        raise FileNotFoundError(
            f"No se encuentra el CSV de entrenamiento: {epochs_info_path}"
        )
    losses, best_epoch = load_losses_from_csv(epochs_info_path)
    selected_epochs = sorted(set([0, best_epoch, epochs]))
    plots_root = os.path.join(model_path, "plots")
    os.makedirs(plots_root, exist_ok=True)
    for threshold_folder in sorted(os.listdir(graph_path)):
        threshold_path = os.path.join(graph_path, threshold_folder)
        if not os.path.isdir(threshold_path):
            continue
        if not threshold_folder.startswith("threshold_"):
            continue
        degree_csv = os.path.join(threshold_path, "degree_index.csv")
        clustering_csv = os.path.join(threshold_path, "clustering_index.csv")
        degree_metrics_csv = os.path.join(threshold_path, "degree_index_metrics.csv")
        clustering_metrics_csv = os.path.join(
            threshold_path, "clustering_index_metrics.csv"
        )
        if os.path.isfile(degree_metrics_csv) and os.path.isfile(
            clustering_metrics_csv
        ):
            save_training_metrics_plot(
                epoch_info_path=epochs_info_path,
                degree_metrics_path=degree_metrics_csv,
                clustering_metrics_path=clustering_metrics_csv,
                output_path=os.path.join(
                    plots_root,
                    threshold_folder,
                    "training_metrics",
                    "training_metrics.pdf",
                ),
                best_epoch=best_epoch,
            )
        if not os.path.isfile(degree_csv) and not os.path.isfile(clustering_csv):
            continue
        degree_df = (
            pd.read_csv(degree_csv) if os.path.isfile(degree_csv) else pd.DataFrame()
        )
        clustering_df = (
            pd.read_csv(clustering_csv)
            if os.path.isfile(clustering_csv)
            else pd.DataFrame()
        )
        partitions = set()
        if not degree_df.empty and "partition" in degree_df.columns:
            partitions.update(degree_df["partition"].dropna().astype(int).tolist())
        if not clustering_df.empty and "partition" in clustering_df.columns:
            partitions.update(clustering_df["partition"].dropna().astype(int).tolist())
        for partition_number in sorted(partitions):
            partition_path = os.path.join(
                plots_root,
                threshold_folder,
                f"partition_{partition_number}",
            )
            os.makedirs(partition_path, exist_ok=True)
            partition_degree_df = (
                degree_df[degree_df["partition"] == partition_number].copy()
                if not degree_df.empty
                else pd.DataFrame()
            )
            partition_clustering_df = (
                clustering_df[clustering_df["partition"] == partition_number].copy()
                if not clustering_df.empty
                else pd.DataFrame()
            )
            save_partition_pdf_plots(
                partition_path=partition_path,
                partition_number=partition_number,
                epochs=selected_epochs,
                losses=losses,
                degree_data=partition_degree_df,
                clustering_data=partition_clustering_df,
            )
