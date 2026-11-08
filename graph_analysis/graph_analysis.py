import os
import pandas as pd
from .plot_utils import (
    save_partition_pdf_plots,
    load_losses_from_csv,
    save_training_metrics_plot,
    save_weighted_graph_training_metrics,
)


def create_separated_panels(model_path, graph_path, logs_path, epochs):
    epochs_info_path = os.path.join(graph_path, "epochs_info.csv")
    if not os.path.isfile(epochs_info_path):
        raise FileNotFoundError(
            f"No se encuentra el CSV de entrenamiento: {epochs_info_path}"
        )
    losses, best_epoch = load_losses_from_csv(epochs_info_path)
    selected_epochs = sorted(set([0, best_epoch, epochs]))
    weighted_graph_csv = os.path.join(graph_path, "weighted_graph")
    strength_csv = os.path.join(weighted_graph_csv, "strength", "strength_index.csv")
    clustering_csv = os.path.join(
        weighted_graph_csv, "clustering", "weighted_clustering_index.csv"
    )
    strength_metrics_csv = os.path.join(
        weighted_graph_csv, "strength", "strength_metrics.csv"
    )
    clustering_metrics_csv = os.path.join(weighted_graph_csv, "clustering","weighted_clustering_metrics.csv")
    strength_df = (
        pd.read_csv(strength_csv) if os.path.isfile(strength_csv) else pd.DataFrame()
    )
    weighted_clustering_df = (
        pd.read_csv(clustering_csv)
        if os.path.isfile(clustering_csv)
        else pd.DataFrame()
    )
    plots_root = os.path.join(model_path, "plots")
    os.makedirs(plots_root, exist_ok=True)
    for threshold_folder in sorted(os.listdir(graph_path)):
        threshold_path = os.path.join(graph_path, threshold_folder)
        if not os.path.isdir(threshold_path):
            continue
        if not (
            threshold_folder.startswith("threshold_")
            or threshold_folder.startswith("backbone_alpha_")
        ):
            continue
        degree_csv = os.path.join(threshold_path, "degree_index.csv")
        clustering_csv = os.path.join(threshold_path, "clustering_index.csv")
        degree_metrics_csv = os.path.join(threshold_path, "degree_index_metrics.csv")
        threshold_clustering_metrics_csv = os.path.join(
            threshold_path, "clustering_index_metrics.csv"
        )
        if os.path.isfile(degree_metrics_csv) and os.path.isfile(
            threshold_clustering_metrics_csv
        ):
            save_training_metrics_plot(
                epoch_info_path=epochs_info_path,
                degree_metrics_path=degree_metrics_csv,
                clustering_metrics_path=threshold_clustering_metrics_csv,
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
    weighted_graph_plots = os.path.join(plots_root, "weighted_graph")
    hidden_layers = set()
    if not strength_df.empty:
        hidden_layers.update(strength_df["hidden_layer"].dropna().astype(int).unique())
    if not weighted_clustering_df.empty:
        hidden_layers.update(
            weighted_clustering_df["hidden_layer"].dropna().astype(int).unique()
        )
    for hidden_layer in sorted(hidden_layers):
        hidden_layer_data = (
            strength_df[strength_df["hidden_layer"] == hidden_layer].copy()
            if not strength_df.empty
            else pd.DataFrame()
        )
        hidden_layer_clustering_data = (
            weighted_clustering_df[
                weighted_clustering_df["hidden_layer"] == hidden_layer
            ].copy()
            if not weighted_clustering_df.empty
            else pd.DataFrame()
        )
        if not hidden_layer_data.empty:
            strength_plot_path = os.path.join(
                weighted_graph_plots,
                "strength",
                f"hidden_layer_{hidden_layer}",
            )
            save_partition_pdf_plots(
                partition_path=strength_plot_path,
                partition_number=hidden_layer,
                epochs=selected_epochs,
                losses=losses,
                strength_data=hidden_layer_data,
                hidden_layer=hidden_layer,
            )
        if not hidden_layer_clustering_data.empty:
            clustering_plot_path = os.path.join(
                weighted_graph_plots,
                "clustering",
                f"hidden_layer_{hidden_layer}",
            )
            save_partition_pdf_plots(
                partition_path=clustering_plot_path,
                partition_number=hidden_layer,
                epochs=selected_epochs,
                losses=losses,
                weighted_clustering_data=hidden_layer_clustering_data,
                hidden_layer=hidden_layer,
            )

    if os.path.isfile(strength_metrics_csv) or os.path.isfile(clustering_metrics_csv):
        training_metrics_path = os.path.join(
            weighted_graph_plots,
            "training_metrics",
        )

        print("strength_metrics_csv:", strength_metrics_csv)
        print("clustering_metrics_csv:", clustering_metrics_csv)
        print("weighted_graph_csv:", weighted_graph_csv)
        save_weighted_graph_training_metrics(
            epoch_info_path=epochs_info_path,
            strength_metrics_path=(
                strength_metrics_csv if os.path.isfile(strength_metrics_csv) else None
            ),
            clustering_metrics_path=(
                clustering_metrics_csv
                if os.path.isfile(clustering_metrics_csv)
                else None
            ),
            output_path=training_metrics_path,
            best_epoch=best_epoch,
        )
