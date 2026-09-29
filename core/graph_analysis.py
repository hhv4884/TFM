import os
import csv
import torch
import torch.nn as nn
from .models import MLP
import matplotlib.pyplot as plt


def degree_distribution(degrees):
    degrees = degrees.detach().cpu()
    unique_degrees, counts = torch.unique(degrees, return_counts=True)
    probabilities = counts.float() / len(degrees)
    return unique_degrees, probabilities


def analyze_partition(weight):
    weight = weight.detach().cpu()
    threshold = torch.median(weight).item()
    #valor absoluto
    mask = weight > threshold
    selected_weights = weight[mask]
    num_connections = mask.sum().item()
    min_weight = weight.min().item()
    max_weight = weight.max().item()

    if selected_weights.numel() > 0:
        min_selected_weight = selected_weights.min().item()
        max_selected_weight = selected_weights.max().item()
    else:
        min_selected_weight = None
        max_selected_weight = None

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
        "min_selected_weight": min_selected_weight,
        "max_selected_weight": max_selected_weight,
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
        txt.write(f"Minimum selected weight: " f"{analysis['min_selected_weight']}\n")
        txt.write(f"Maximum selected weight: " f"{analysis['max_selected_weight']}\n")
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
                    "min_selected_weight",
                    "max_selected_weight",
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
                    analysis["min_selected_weight"],
                    analysis["max_selected_weight"],
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
                    analysis["min_selected_weight"],
                    analysis["max_selected_weight"],
                ]
            )


def save_graph_analysis(model, path, epoch):
    layer_number = 0
    for layer in model.net:
        if not isinstance(layer, nn.Linear):
            continue
        partition_number = layer_number + 1
        partition_path = os.path.join(path, f"partition_{partition_number}")
        os.makedirs(partition_path, exist_ok=True)
        analysis = analyze_partition(layer.weight)
        save_degree_histograms(analysis, partition_path, epoch, partition_number)
        save_graph_txt(analysis, partition_path, epoch, partition_number, layer_number)
        save_graph_csv(analysis, partition_path, epoch, partition_number, layer_number)
        layer_number += 1
