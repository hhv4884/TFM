import os
import csv
import pandas as pd


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
    analyze_all_partitions(model, epoch, thresholds, csv_path, calculate_strength)


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
    epoch, hidden_layer, input_strength, output_strength, total_strength, strength_path
):
    os.makedirs(strength_path, exist_ok=True)
    csv_file = os.path.join(strength_path, "strength_index.csv")
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
            }
        )
    df = pd.DataFrame(data)
    if os.path.isfile(csv_file):
        df.to_csv(csv_file, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_file, index=False)
