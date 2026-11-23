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
    backbone_alphas=(0.01,),
    calculate_disparity=False,
    disparity_alphas=(0.05,),
    disparity_both_ends=True,
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
        backbone_alphas,
        calculate_disparity,
        disparity_alphas,
        disparity_both_ends,
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
    partition,
    input_strength,
    output_strength,
    strength_path,
):
    os.makedirs(strength_path, exist_ok=True)
    values_csv = os.path.join(strength_path, "strength_index.csv")
    metrics_csv = os.path.join(strength_path, "strength_metrics.csv")
    input_strength = input_strength.detach().cpu().numpy()
    output_strength = output_strength.detach().cpu().numpy()
    rows = []
    for neuron_type, values in [
        ("in", input_strength),
        ("out", output_strength),
    ]:
        n_nodes = len(values)
        for neuron_index, value in enumerate(values):
            rows.append(
                {
                    "epoch": epoch,
                    "partition": partition,
                    "neuron_type": neuron_type,
                    "neuron_index": neuron_index,
                    "strength": float(value),
                    "n_nodes": n_nodes
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(
        values_csv,
        mode="a" if os.path.isfile(values_csv) else "w",
        header=not os.path.isfile(values_csv),
        index=False,
    )
    metrics_rows = []
    for neuron_type, values in [
        ("in", input_strength),
        ("out", output_strength),
    ]:
        valid_values = values[np.isfinite(values)]
        metrics_rows.append(
            {
                "epoch": epoch,
                "partition": partition,
                "neuron_type": neuron_type,
                "mean": float(np.mean(valid_values)) if len(valid_values) else np.nan,
                "median": (
                    float(np.median(valid_values)) if len(valid_values) else np.nan
                ),
                "std_dev": float(np.std(valid_values)) if len(valid_values) else np.nan,
                "n_valid": len(valid_values),
                "n_nodes": len(values)
            }
        )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(
        metrics_csv,
        mode="a" if os.path.isfile(metrics_csv) else "w",
        header=not os.path.isfile(metrics_csv),
        index=False,
    )


def update_csv_weighted_clustering_info(
    epoch,
    partition,
    neuron_type,
    coefficients,
    mean,
    median,
    std_dev,
    clustering_path,
    n_edges,
    n_nodes,
):
    os.makedirs(clustering_path, exist_ok=True)
    values_csv = os.path.join(
        clustering_path,
        "weighted_clustering_index.csv",
    )
    metrics_csv = os.path.join(
        clustering_path,
        "weighted_clustering_metrics.csv",
    )
    rows = []
    for neuron_index, coefficient in enumerate(coefficients):
        rows.append(
            {
                "epoch": epoch,
                "partition": partition,
                "neuron_type": neuron_type,
                "neuron_index": neuron_index,
                "clustering_coefficient": float(coefficient),
                "n_edges": n_edges,
                "n_nodes": n_nodes,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(
        values_csv,
        mode="a" if os.path.isfile(values_csv) else "w",
        header=not os.path.isfile(values_csv),
        index=False,
    )
    metrics = pd.DataFrame(
        [
            {
                "epoch": epoch,
                "partition": partition,
                "neuron_type": neuron_type,
                "mean": mean,
                "median": median,
                "std_dev": std_dev,
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


def update_csv_disparity_info(epoch, partition, keep, disparity_path):
    os.makedirs(disparity_path, exist_ok=True)
    keep = keep.detach().cpu().bool()
    in_degree = keep.sum(dim=0).numpy()
    out_degree = keep.sum(dim=1).numpy()
    n_edges = int(keep.sum().item())
    n_outputs, n_inputs = keep.shape
    n_possible_edges = n_outputs * n_inputs
    density = n_edges / n_possible_edges if n_possible_edges > 0 else 0.0
    values_path = os.path.join(disparity_path, "disparity_index.csv")
    rows = []
    for neuron_index, degree in enumerate(in_degree):
        rows.append(
            {
                "epoch": epoch,
                "partition": partition,
                "neuron_type": "in",
                "neuron_index": neuron_index,
                "degree": int(degree),
            }
        )
    for neuron_index, degree in enumerate(out_degree):
        rows.append(
            {
                "epoch": epoch,
                "partition": partition,
                "neuron_type": "out",
                "neuron_index": neuron_index,
                "degree": int(degree),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(
        values_path,
        mode="a" if os.path.isfile(values_path) else "w",
        header=not os.path.isfile(values_path),
        index=False,
    )
    metrics_path = os.path.join(
        disparity_path,
        "disparity_metrics.csv",
    )
    metrics = pd.DataFrame(
        [
            {
                "epoch": epoch,
                "partition": partition,
                "n_edges": n_edges,
                "n_possible_edges": n_possible_edges,
                "density": density,
                "mean_in_degree": float(np.mean(in_degree)),
                "mean_out_degree": float(np.mean(out_degree)),
            }
        ]
    )
    metrics.to_csv(
        metrics_path,
        mode="a" if os.path.isfile(metrics_path) else "w",
        header=not os.path.isfile(metrics_path),
        index=False,
    )
