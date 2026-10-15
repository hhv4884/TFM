import os
import torch.nn as nn
from .metrics import analyze_partition, bip_clustering
from .csv_utils import (
    save_degree_distribution_csv,
    save_clustering_csv,
)
from .report_utils import (
    get_best_epoch_from_logs,
    get_epoch_label,
    save_degree_txt,
    save_clustering_txt,
)
from .plot_utils import save_partition_svg_plots


def save_graph_analysis(model, path, epoch, best_epoch, last_epoch, threshold=None):
    for layer_number, layer in enumerate(
        [layer for layer in model.net if isinstance(layer, nn.Linear)]
    ):
        partition_number = layer_number + 1
        partition_path = os.path.join(path, f"partition_{partition_number}")
        os.makedirs(partition_path, exist_ok=True)
        analysis = analyze_partition(layer.weight, threshold=threshold, threshold_mode="percentile")
        save_degree_distribution_csv(
            analysis,
            partition_path,
            epoch,
            partition_number,
        )
        epoch_label = get_epoch_label(
            epoch,
            best_epoch,
            last_epoch,
        )
        save_degree_txt(
            analysis,
            partition_path,
            epoch,
            epoch_label,
            partition_number,
        )
        A = analysis["mask"].float()
        cc_in = bip_clustering(A)
        cc_out = bip_clustering(A.T)
        save_clustering_csv(
            cc_in,
            cc_out,
            partition_path,
            epoch,
            partition_number,
        )
        save_clustering_txt(
            cc_in,
            cc_out,
            partition_path,
            epoch,
            epoch_label,
            partition_number,
        )


def analyze_model_with_thresholds(
    model, graph_path, epoch, best_epoch, last_epoch, thresholds=None
):
    if thresholds is None:
        thresholds = [None]
    elif isinstance(thresholds, (int, float)):
        thresholds = [thresholds]
    for threshold in thresholds:
        if threshold is None:
            threshold_path = os.path.join(graph_path, "threshold_median")
        else:
            threshold_path = os.path.join(graph_path, f"threshold_{threshold}")
        os.makedirs(threshold_path, exist_ok=True)
        save_graph_analysis(
            model,
            threshold_path,
            epoch,
            best_epoch,
            last_epoch,
            threshold,
        )


def create_separated_panels(model_path, graph_path, logs_path, epochs):
    best_epoch = get_best_epoch_from_logs(logs_path)
    last_epoch = epochs - 1
    selected_epochs = sorted(set([0, best_epoch, last_epoch]))
    for threshold_folder in os.listdir(graph_path):
        threshold_path = os.path.join(graph_path, threshold_folder)
        if not os.path.isdir(threshold_path):
            continue
        for folder in os.listdir(threshold_path):
            if not folder.startswith("partition_"):
                continue
            partition_path = os.path.join(threshold_path, folder)
            if not os.path.isdir(partition_path):
                continue
            try:
                partition_number = int(folder.split("_")[1])
            except (IndexError, ValueError):
                continue
            save_partition_svg_plots(partition_path, partition_number, selected_epochs)
