import os
import re
import csv
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt


def degree_distribution(degrees):
    degrees = degrees.detach().cpu()
    unique_degrees, counts = torch.unique(degrees, return_counts=True)
    probabilities = counts.float() / len(degrees)
    return unique_degrees, probabilities


def analyze_partition(weight, threshold=None):
    weight = weight.detach().cpu()
    if threshold is None:
        threshold = torch.median(torch.abs(weight)).item()
    mask = torch.abs(weight) > threshold
    num_connections = mask.sum().item()
    min_weight = weight.min().item()
    max_weight = weight.max().item()

    in_degree = mask.sum(dim=1)
    out_degree = mask.sum(dim=0)

    in_degree_values, in_degree_probabilities = degree_distribution(in_degree)
    out_degree_values, out_degree_probabilities = degree_distribution(out_degree)

    return {
        "threshold": threshold,
        "mask": mask,
        "num_connections": num_connections,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "in_degree_values": in_degree_values,
        "in_degree_probabilities": in_degree_probabilities,
        "out_degree_values": out_degree_values,
        "out_degree_probabilities": out_degree_probabilities,
    }


def save_degree_histograms(analysis, path, epoch, partition_number):
    plt.figure()
    in_degrees = analysis["in_degree_values"].numpy()
    in_probabilities = analysis["in_degree_probabilities"].numpy()
    plt.bar(in_degrees, in_probabilities, width=0.8, align="center")
    plt.xlabel("In-degree")
    plt.ylabel("Probability")
    plt.title(
        f"Partition {partition_number} - " f"In-degree distribution - Epoch {epoch}"
    )
    in_degree_path = os.path.join(path, f"epoch_{epoch}_in_degree.png")
    plt.savefig(in_degree_path, dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure()
    out_degrees = analysis["out_degree_values"].numpy()
    out_probabilities = analysis["out_degree_probabilities"].numpy()
    plt.bar(out_degrees, out_probabilities, width=0.8, align="center")
    plt.xlabel("Out-degree")
    plt.ylabel("Probability")
    plt.title(
        f"Partition {partition_number} - " f"Out-degree distribution - Epoch {epoch}"
    )
    out_degree_path = os.path.join(path, f"epoch_{epoch}_out_degree.png")
    plt.savefig(out_degree_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_graph_txt(analysis, path, epoch, partition_number, layer_number):
    txt_file = os.path.join(path, f"epoch_{epoch}_graph.txt")
    with open(txt_file, "w") as txt:
        txt.write(f"Epoch: {epoch}\n")
        txt.write(f"Partition: {partition_number}\n")
        txt.write(f"Connection: " f"L{layer_number} -> " f"L{layer_number + 1}\n")
        txt.write("\n")
        txt.write(
            f"-.-.-.-.-.- "
            f"PARTITION {partition_number} "
            f"(L{layer_number} -> "
            f"L{layer_number + 1}) "
            f"-.-.-.-.-.-\n\n"
        )
        txt.write(f"Number of connections: " f"{analysis['num_connections']}\n")
        txt.write(f"Minimum weight: " f"{analysis['min_weight']:.6f}\n")
        txt.write(f"Maximum weight: " f"{analysis['max_weight']:.6f}\n")
        txt.write(f"Threshold: " f"{analysis['threshold']:.6f}\n")
        txt.write("\n" "-.-.-.-.- IN-DEGREE -.-.-.-.-\n")
        for degree, probability in zip(
            analysis["in_degree_values"], analysis["in_degree_probabilities"]
        ):
            txt.write(
                f"Degree {degree.item()} -> " f"Probability {probability.item():.6f}\n"
            )
        txt.write("\n" "-.-.-.-.- OUT-DEGREE -.-.-.-.-\n")
        for degree, probability in zip(
            analysis["out_degree_values"], analysis["out_degree_probabilities"]
        ):
            txt.write(
                f"Degree {degree.item()} -> " f"Probability {probability.item():.6f}\n"
            )
    return txt_file


def save_graph_csv(analysis, path, epoch, partition_number, layer_number):
    csv_file = os.path.join(path, "graph_analysis.csv")
    csv_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as csv_file_object:
        csv_writer = csv.writer(csv_file_object)
        if not csv_exists:
            csv_writer.writerow(
                [
                    "epoch",
                    "partition",
                    "layer_origin",
                    "layer_target",
                    "threshold",
                    "degree_type",
                    "degree",
                    "probability",
                    "num_connections",
                    "min_weight",
                    "max_weight",
                ]
            )
        for degree, probability in zip(
            analysis["in_degree_values"], analysis["in_degree_probabilities"]
        ):
            csv_writer.writerow(
                [
                    epoch,
                    partition_number,
                    layer_number,
                    layer_number + 1,
                    analysis["threshold"],
                    "in",
                    degree.item(),
                    probability.item(),
                    analysis["num_connections"],
                    analysis["min_weight"],
                    analysis["max_weight"],
                ]
            )
        for degree, probability in zip(
            analysis["out_degree_values"], analysis["out_degree_probabilities"]
        ):
            csv_writer.writerow(
                [
                    epoch,
                    partition_number,
                    layer_number,
                    layer_number + 1,
                    analysis["threshold"],
                    "out",
                    degree.item(),
                    probability.item(),
                    analysis["num_connections"],
                    analysis["min_weight"],
                    analysis["max_weight"],
                ]
            )


def save_graph_analysis(model, path, epoch, threshold=None):
    layer_number = 0
    for layer in model.net:
        if not isinstance(layer, nn.Linear):
            continue
        partition_number = layer_number + 1
        partition_path = os.path.join(path, f"partition_{partition_number}")
        os.makedirs(partition_path, exist_ok=True)
        analysis = analyze_partition(layer.weight, threshold=threshold)
        save_degree_histograms(analysis, partition_path, epoch, partition_number)
        save_graph_txt(analysis, partition_path, epoch, partition_number, layer_number)
        save_graph_csv(analysis, partition_path, epoch, partition_number, layer_number)
        layer_number += 1


def get_best_epoch_from_logs(logs_path):
    best_epoch = None
    if not os.path.exists(logs_path):
        raise FileNotFoundError(f"Logs path not found: {logs_path}")
    for filename in os.listdir(logs_path):
        if not filename.startswith("log_epoch_"):
            continue
        if not filename.endswith(".txt"):
            continue
        log_file = os.path.join(logs_path, filename)
        with open(log_file, "r") as f:
            content = f.read()
        matches = re.findall(r"BEST EPOCH:\s*(\d+)", content)
        if matches:
            epoch = int(matches[-1]) - 1
            if best_epoch is None or epoch > best_epoch:
                best_epoch = epoch
    if best_epoch is None:
        raise ValueError(f"Could not find 'BEST EPOCH' in logs located at {logs_path}")
    return best_epoch


def save_final_weight_panels(model_path, graph_path, logs_path, epochs):
    final_panel_path = os.path.join(model_path, "final_panel")
    os.makedirs(final_panel_path, exist_ok=True)
    best_epoch = get_best_epoch_from_logs(logs_path)
    for threshold_folder in os.listdir(graph_path):
        threshold_path = os.path.join(
            graph_path,
            threshold_folder,
        )
        if not os.path.isdir(threshold_path):
            continue
        threshold_panel_path = os.path.join(
            final_panel_path,
            threshold_folder,
        )
        os.makedirs(threshold_panel_path, exist_ok=True)
        partition_numbers = []
        for folder in os.listdir(threshold_path):
            folder_path = os.path.join(
                threshold_path,
                folder,
            )
            if os.path.isdir(folder_path) and folder.startswith("partition_"):
                try:
                    partition_number = int(folder.split("_")[1])
                    partition_numbers.append(partition_number)
                except (IndexError, ValueError):
                    continue
        partition_numbers.sort()
        for partition_number in partition_numbers:
            partition_path = os.path.join(
                threshold_path,
                f"partition_{partition_number}",
            )
            csv_path = os.path.join(
                partition_path,
                "graph_analysis.csv",
            )
            if not os.path.exists(csv_path):
                print(
                    f"Skipping partition {partition_number}: "
                    f"CSV not found at {csv_path}"
                )
                continue
            data = pd.read_csv(csv_path)
            epoch_0_rows_in = data[
                (data["epoch"] == 0) & (data["degree_type"] == "in")
            ][["degree", "probability"]]
            epoch_0_rows_out = data[
                (data["epoch"] == 0) & (data["degree_type"] == "out")
            ][["degree", "probability"]]
            epoch_best_rows_in = data[
                (data["epoch"] == best_epoch) & (data["degree_type"] == "in")
            ][["degree", "probability"]]
            epoch_best_rows_out = data[
                (data["epoch"] == best_epoch) & (data["degree_type"] == "out")
            ][["degree", "probability"]]
            last_epoch = epochs - 1
            epoch_last_rows_in = data[
                (data["epoch"] == last_epoch) & (data["degree_type"] == "in")
            ][["degree", "probability"]]
            epoch_last_rows_out = data[
                (data["epoch"] == last_epoch) & (data["degree_type"] == "out")
            ][["degree", "probability"]]
            for orientation in ["vertical", "horizontal"]:
                if orientation == "vertical":
                    fig, axes = plt.subplots(3, 1, figsize=(8, 15))
                else:
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                axes[0].plot(
                    epoch_0_rows_in["degree"],
                    epoch_0_rows_in["probability"],
                    marker="o",
                    label="In-degree",
                )
                axes[0].plot(
                    epoch_0_rows_out["degree"],
                    epoch_0_rows_out["probability"],
                    marker="o",
                    label="Out-degree",
                )
                axes[0].set_title("Epoch 0")
                axes[0].set_xlabel("Degree")
                axes[0].set_ylabel("Probability")
                axes[0].grid(alpha=0.3)
                axes[0].legend()
                axes[1].plot(
                    epoch_best_rows_in["degree"],
                    epoch_best_rows_in["probability"],
                    marker="o",
                    label="In-degree",
                )
                axes[1].plot(
                    epoch_best_rows_out["degree"],
                    epoch_best_rows_out["probability"],
                    marker="o",
                    label="Out-degree",
                )
                axes[1].set_title(f"Best epoch ({best_epoch})")
                axes[1].set_xlabel("Degree")
                axes[1].set_ylabel("Probability")
                axes[1].grid(alpha=0.3)
                axes[1].legend()
                axes[2].plot(
                    epoch_last_rows_in["degree"],
                    epoch_last_rows_in["probability"],
                    marker="o",
                    label="In-degree",
                )
                axes[2].plot(
                    epoch_last_rows_out["degree"],
                    epoch_last_rows_out["probability"],
                    marker="o",
                    label="Out-degree",
                )
                axes[2].set_title(f"Last epoch ({last_epoch})")
                axes[2].set_xlabel("Degree")
                axes[2].set_ylabel("Probability")
                axes[2].grid(alpha=0.3)
                axes[2].legend()
                fig.suptitle(
                    f"{threshold_folder} - "
                    f"Partition {partition_number} - "
                    f"Degree Distribution",
                    fontsize=16,
                )
                fig.tight_layout()
                panel_path = os.path.join(
                    threshold_panel_path,
                    f"partition_{partition_number}_{orientation}.png",
                )
                fig.savefig(panel_path, dpi=300, bbox_inches="tight")
                plt.close(fig)


def analyze_model_with_thresholds(model, graph_path, epoch, thresholds=None):
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
        save_graph_analysis(model, threshold_path, epoch=epoch, threshold=threshold)


def create_separated_panels(model_path, graph_path, logs_path, epochs):
    final_panel_path = os.path.join(model_path, "final_panel_separated")
    os.makedirs(final_panel_path, exist_ok=True)
    best_epoch = get_best_epoch_from_logs(logs_path)
    last_epoch = epochs - 1
    print(f"\nProcesando: {model_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Last epoch: {last_epoch}")
    for threshold_folder in os.listdir(graph_path):
        threshold_path = os.path.join(graph_path, threshold_folder)
        if not os.path.isdir(threshold_path):
            continue
        threshold_panel_path = os.path.join(
            final_panel_path,
            threshold_folder)
        os.makedirs(threshold_panel_path, exist_ok=True)
        partition_numbers = []
        for folder in os.listdir(threshold_path):
            folder_path = os.path.join(threshold_path, folder)
            if os.path.isdir(folder_path) and folder.startswith("partition_"):
                try:
                    partition_number = int(folder.split("_")[1])
                    partition_numbers.append(partition_number)
                except (IndexError, ValueError):
                    continue
        partition_numbers.sort()
        for partition_number in partition_numbers:
            partition_path = os.path.join(
                threshold_path,
                f"partition_{partition_number}")
            csv_path = os.path.join(
                partition_path,
                "graph_analysis.csv")
            if not os.path.exists(csv_path):
                print(f"CSV no encontrado: {csv_path}")
                continue
            data = pd.read_csv(csv_path)
            epochs_data = []
            for epoch in [0, best_epoch, last_epoch]:
                rows = data[
                    (data["epoch"] == epoch)
                    & (data["degree_type"] == "in")
                ][["degree", "probability"]]
                epochs_data.append((epoch, rows))
            fig, axes = plt.subplots(
                3,
                1,
                figsize=(8, 15))
            for ax, (epoch, rows) in zip(axes, epochs_data):
                ax.bar(
                    rows["degree"],
                    rows["probability"],
                    width=0.8,
                    align="center")
                if epoch == 0:
                    title = "Epoch 0"
                elif epoch == best_epoch:
                    title = f"Best epoch ({best_epoch})"
                else:
                    title = f"Last epoch ({last_epoch})"
                ax.set_title(title)
                ax.set_xlabel("In-degree")
                ax.set_ylabel("Probability")
                ax.grid(alpha=0.3)
            fig.suptitle(
                f"{threshold_folder} - "
                f"Partition {partition_number} - "
                f"In-degree Distribution",
                fontsize=16)
            fig.tight_layout()
            path = os.path.join(
                threshold_panel_path,
                f"partition_{partition_number}_in_vertical.png")
            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight")
            plt.close(fig)
            fig, axes = plt.subplots(
                1,
                3,
                figsize=(18, 5))
            for ax, (epoch, rows) in zip(axes, epochs_data):
                ax.bar(
                    rows["degree"],
                    rows["probability"],
                    width=0.8,
                    align="center")
                if epoch == 0:
                    title = "Epoch 0"
                elif epoch == best_epoch:
                    title = f"Best epoch ({best_epoch})"
                else:
                    title = f"Last epoch ({last_epoch})"
                ax.set_title(title)
                ax.set_xlabel("In-degree")
                ax.set_ylabel("Probability")
                ax.grid(alpha=0.3)
            fig.suptitle(
                f"{threshold_folder} - "
                f"Partition {partition_number} - "
                f"In-degree Distribution",
                fontsize=16)
            fig.tight_layout()
            path = os.path.join(
                threshold_panel_path,
                f"partition_{partition_number}_in_horizontal.png")
            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight")
            plt.close(fig)
            epochs_data = []
            for epoch in [0, best_epoch, last_epoch]:
                rows = data[
                    (data["epoch"] == epoch)
                    & (data["degree_type"] == "out")
                ][["degree", "probability"]]
                epochs_data.append((epoch, rows))
            fig, axes = plt.subplots(
                3,
                1,
                figsize=(8, 15))
            for ax, (epoch, rows) in zip(axes, epochs_data):
                ax.bar(
                    rows["degree"],
                    rows["probability"],
                    width=0.8,
                    align="center")
                if epoch == 0:
                    title = "Epoch 0"
                elif epoch == best_epoch:
                    title = f"Best epoch ({best_epoch})"
                else:
                    title = f"Last epoch ({last_epoch})"
                ax.set_title(title)
                ax.set_xlabel("Out-degree")
                ax.set_ylabel("Probability")
                ax.grid(alpha=0.3)
            fig.suptitle(
                f"{threshold_folder} - "
                f"Partition {partition_number} - "
                f"Out-degree Distribution",
                fontsize=16)
            fig.tight_layout()
            path = os.path.join(
                threshold_panel_path,
                f"partition_{partition_number}_out_vertical.png")
            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight")
            plt.close(fig)
            fig, axes = plt.subplots(
                1,
                3,
                figsize=(18, 5))
            for ax, (epoch, rows) in zip(axes, epochs_data):
                ax.bar(
                    rows["degree"],
                    rows["probability"],
                    width=0.8,
                    align="center")
                if epoch == 0:
                    title = "Epoch 0"
                elif epoch == best_epoch:
                    title = f"Best epoch ({best_epoch})"
                else:
                    title = f"Last epoch ({last_epoch})"
                ax.set_title(title)
                ax.set_xlabel("Out-degree")
                ax.set_ylabel("Probability")
                ax.grid(alpha=0.3)
            fig.suptitle(
                f"{threshold_folder} - "
                f"Partition {partition_number} - "
                f"Out-degree Distribution",
                fontsize=16)
            fig.tight_layout()
            path = os.path.join(
                threshold_panel_path,
                f"partition_{partition_number}_out_horizontal.png")
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)

