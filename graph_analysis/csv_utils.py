import os
import pandas as pd


def save_degree_distribution_csv(analysis, path, epoch, partition_number):
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, "degree_distribution.csv")
    rows = []
    for degree_type in ("in", "out"):
        values = analysis[f"{degree_type}_degree_values"]
        counts = analysis[f"{degree_type}_degree_counts"]
        probabilities = analysis[f"{degree_type}_degree_probabilities"]
        for degree, count, probability in zip(
            values.tolist(),
            counts.tolist(),
            probabilities.tolist()):
            rows.append({
                "epoch": epoch,
                "partition": partition_number,
                "degree_type": degree_type,
                "degree": degree,
                "count": count,
                "probability": probability,
            })
    df = pd.DataFrame(rows)
    _append_without_duplicate_epoch(
        df, csv_path, epoch, ["epoch", "partition", "degree_type", "degree"])


def save_clustering_csv(cc_in, cc_out, path, epoch, partition_number):
    os.makedirs(path, exist_ok=True)
    csv_path = os.path.join(path, "clustering_coefficient.csv")
    rows = []
    for neuron_type, coefficients in (("in", cc_in), ("out", cc_out)):
        for neuron_index, coefficient in enumerate(coefficients):
            rows.append({
                "epoch": epoch,
                "partition": partition_number,
                "neuron_type": neuron_type,
                "neuron_index": neuron_index,
                "clustering_coefficient": float(coefficient)})
    df = pd.DataFrame(rows)
    _append_without_duplicate_epoch(
        df,
        csv_path,
        epoch,
        ["epoch", "partition", "neuron_type", "neuron_index"])

def _append_without_duplicate_epoch(new_df, csv_path, epoch, key_columns):
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        old_df = old_df[old_df["epoch"] != epoch]
        df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        df = new_df
    df = df.sort_values(key_columns)
    df.to_csv(csv_path, index=False)
