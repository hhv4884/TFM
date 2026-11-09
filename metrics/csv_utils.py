import os
import csv
import pandas as pd
import numpy as np


def update_csv_epoch_info(
    model,
    epoch,
    train_loss,
    train_accuracy,
    test_loss,
    test_accuracy,
    csv_path,
    thresholds,
    calculate_strength=False,
    calculate_weighted_clustering=False,
    backbone_alpha=0.01,
):
    from .analysis import analyze_all_partitions

    os.makedirs(csv_path, exist_ok=True)
    epoch_info = os.path.join(csv_path, "epochs_info.csv")
    if not os.path.exists(epoch_info):
        with open(epoch_info, "w", encoding="utf-8") as f:
            epoch_info_columns = [
                "epoch",
                "train_loss",
                "train_accuracy",
                "test_loss",
                "test_accuracy",
            ]
            first_row_info = [0, train_loss, train_accuracy, test_loss, test_accuracy]
            writer = csv.writer(f)
            writer.writerow(epoch_info_columns)
            writer.writerow(first_row_info)
    else:
        with open(epoch_info, "a", newline="", encoding="utf-8") as f:
            new_row = [epoch, train_loss, train_accuracy, test_loss, test_accuracy]
            writer = csv.writer(f)
            writer.writerow(new_row)
    analyze_all_partitions(
        model,
        epoch,
        thresholds,
        csv_path,
        calculate_strength,
        calculate_weighted_clustering,
        backbone_alpha,
    )


def update_csv_clustering_info(
    epoch,
    partition_number,
    neuron_type,
    clustering_coefficients,
    mean,
    median,
    std_dev,
    clustering_path,
):
    os.makedirs(clustering_path, exist_ok=True)
    clustering_index = os.path.join(clustering_path, "clustering_index.csv")
    clustering_index_metrics = os.path.join(
        clustering_path, "clustering_index_metrics.csv"
    )

    if not os.path.exists(clustering_index):
        with open(clustering_index, "w", encoding="utf-8") as f:
            clustering_info_columns = [
                "epoch",
                "partition",
                "neuron_type",
                "neuron_index",
                "clustering_coefficient",
            ]
            writer = csv.writer(f)
            writer.writerow(clustering_info_columns)

        with open(clustering_index_metrics, "w", encoding="utf-8") as f:
            clustering_metrics_columns = [
                "epoch",
                "partition",
                "neuron_type",
                "mean",
                "median",
                "std_dev",
            ]
            writer = csv.writer(f)
            writer.writerow(clustering_metrics_columns)
            pass
    with open(clustering_index, "a", newline="", encoding="utf-8") as f:
        rows = [
            [
                epoch,
                partition_number,
                neuron_type,
                neuron_index,
                float(coefficient),
            ]
            for neuron_index, coefficient in enumerate(clustering_coefficients)
        ]
        writer = csv.writer(f)
        writer.writerows(rows)

    with open(clustering_index_metrics, "a", newline="", encoding="utf-8") as f:
        row = [epoch, partition_number, neuron_type, mean, median, std_dev]
        writer = csv.writer(f)
        writer.writerow(row)


def update_csv_degree_info(
    epoch,
    partition_number,
    in_degree_values,
    in_degree_count,
    in_degree_probabilities,
    out_degree_values,
    out_degree_count,
    out_degree_probabilities,
    in_mean,
    in_median,
    in_std_dev,
    out_mean,
    out_median,
    out_std_dev,
    degree_path,
):
    os.makedirs(degree_path, exist_ok=True)
    degree_index = os.path.join(degree_path, "degree_index.csv")
    degree_index_metrics = os.path.join(degree_path, "degree_index_metrics.csv")
    if not os.path.exists(degree_index):
        with open(degree_index, "w", encoding="utf-8") as f:
            degree_info_columns = [
                "epoch",
                "partition",
                "neuron_type",
                "degree",
                "count",
                "probability",
            ]
            writer = csv.writer(f)
            writer.writerow(degree_info_columns)

        with open(degree_index_metrics, "w", encoding="utf-8") as f:
            degree_metrics_columns = [
                "epoch",
                "partition",
                "neuron_type",
                "mean",
                "median",
                "std_dev",
            ]
            writer = csv.writer(f)
            writer.writerow(degree_metrics_columns)
            pass
    with open(degree_index, "a", newline="", encoding="utf-8") as f:
        rows = []
        for neuron_type, degrees, counts, probabilities in [
            ("in", in_degree_values, in_degree_count, in_degree_probabilities),
            ("out", out_degree_values, out_degree_count, out_degree_probabilities),
        ]:
            for degree, count, probability in zip(
                degrees.tolist(), counts.tolist(), probabilities.tolist()
            ):
                rows.append(
                    [epoch, partition_number, neuron_type, degree, count, probability]
                )
        writer = csv.writer(f)
        writer.writerows(rows)

    with open(degree_index_metrics, "a", newline="", encoding="utf-8") as f:
        rows = [
            [epoch, partition_number, "in", in_mean, in_median, in_std_dev],
            [epoch, partition_number, "out", out_mean, out_median, out_std_dev],
        ]
        writer = csv.writer(f)
        writer.writerows(rows)


def update_csv_strength_info(
    epoch,
    hidden_layer,
    input_strength,
    output_strength,
    total_strength,
    strength_path,
    n_edges,
    n_nodes,
):
    os.makedirs(strength_path, exist_ok=True)
    values_csv = os.path.join(strength_path, "strength_index.csv")
    metrics_csv = os.path.join(strength_path, "strength_metrics.csv")
    data = []
    for i in range(len(total_strength)):
        data.append(
            {
                "epoch": epoch,
                "hidden_layer": hidden_layer,
                "neuron_index": i,
                "input_strength": input_strength[i].item(),
                "output_strength": output_strength[i].item(),
                "strength": total_strength[i].item(),
                "n_edges": n_edges,
                "n_nodes": n_nodes,
            }
        )
    df = pd.DataFrame(data)
    df.to_csv(
        values_csv,
        mode="a" if os.path.isfile(values_csv) else "w",
        header=not os.path.isfile(values_csv),
        index=False,
    )
    values = total_strength.detach().cpu().numpy()
    values = values[np.isfinite(values)]
    metrics = pd.DataFrame(
        [
            {
                "epoch": epoch,
                "hidden_layer": hidden_layer,
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "median": float(np.median(values)) if len(values) else np.nan,
                "std_dev": float(np.std(values)) if len(values) else np.nan,
                "n_valid": len(values),
                "n_edges": n_edges,
                "n_nodes": n_nodes,
            }
        ]
    )
    metrics.to_csv(
        metrics_csv,
        mode="a" if os.path.isfile(metrics_csv) else "w",
        header=not os.path.isfile(metrics_csv),
        index=False,
    )


def update_csv_weighted_clustering_info(
    epoch, hidden_layer, coefficients, mean, median, std_dev, clustering_path, n_edges, n_nodes
):
    os.makedirs(clustering_path, exist_ok=True)
    values_csv = os.path.join(clustering_path, "weighted_clustering_index.csv")
    metrics_csv = os.path.join(clustering_path, "weighted_clustering_metrics.csv")
    rows = []
    for neuron_index, coefficient in enumerate(coefficients):
        rows.append(
            {
                "epoch": epoch,
                "hidden_layer": hidden_layer,
                "neuron_index": neuron_index,
                "clustering_coefficient": float(coefficient),
                "n_edges": n_edges,
                "n_nodes": n_nodes
            }
        )
    df = pd.DataFrame(rows)
    if os.path.isfile(values_csv):
        df.to_csv(values_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(values_csv, index=False)
    metrics = pd.DataFrame(
        [
            {
                "epoch": epoch,
                "hidden_layer": hidden_layer,
                "mean": mean,
                "median": median,
                "std_dev": std_dev,
                "n_edges": n_edges,
                "n_nodes": n_nodes
            }
        ]
    )
    if os.path.isfile(metrics_csv):
        metrics.to_csv(
            metrics_csv,
            mode="a",
            header=False,
            index=False,
        )
    else:
        metrics.to_csv(metrics_csv, index=False)
