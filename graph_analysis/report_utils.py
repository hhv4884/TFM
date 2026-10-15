import os
import numpy as np
import re


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


def get_epoch_label(epoch, best_epoch, last_epoch):
    if epoch == 0:
        return "epoch_0"
    if epoch == best_epoch:
        return "best_epoch"
    if epoch == last_epoch:
        return "last_epoch"
    return f"epoch_{epoch}"



def save_degree_txt(analysis, path, epoch, epoch_label, partition_number):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(
        path, f"{epoch_label}_degree_distribution.txt"
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Partition: {partition_number}\n")
        f.write(f"Threshold: {analysis['threshold']}\n")
        f.write(f"Number of connections: {analysis['num_connections']}\n")
        f.write(f"Minimum weight: {analysis['min_weight']}\n")
        f.write(f"Maximum weight: {analysis['max_weight']}\n")
        for degree_type in ("in", "out"):
            degrees = analysis[f"{degree_type}_degree"].detach().cpu().numpy().astype(float)
            f.write(f"\n{degree_type.upper()}-DEGREE DISTRIBUTION\n")
            f.write(f"Mean: {np.mean(degrees):.6f}\n")
            f.write(f"Median: {np.median(degrees):.6f}\n")
            f.write(f"Std: {np.std(degrees):.6f}\n")
            values, counts = np.unique(degrees, return_counts=True)
            total = counts.sum()
            f.write("Degree | Count | Probability (%)\n")
            for degree, count in zip(values, counts):
                probability = count / total * 100 if total else 0.0
                f.write(
                    f"{int(degree)} | {int(count)} | {probability:.4f}\n"
                )


def save_clustering_txt(cc_in, cc_out, path, epoch, epoch_label, partition_number):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, f"{epoch_label}_clustering_coefficient.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Partition: {partition_number}\n")
        for neuron_type, coefficients in (("in", cc_in), ("out", cc_out)):
            coefficients = np.asarray(coefficients)
            f.write(f"\n{neuron_type.upper()} CLUSTERING\n")
            f.write(f"Number of nodes: {len(coefficients)}\n")
            if len(coefficients) == 0:
                f.write("No values available.\n")
                continue
            f.write(f"Mean: {coefficients.mean():.6f}\n")
            f.write(f"Median: {np.median(coefficients):.6f}\n")
            f.write(f"Std: {coefficients.std():.6f}\n")
            f.write(f"Min: {coefficients.min():.6f}\n")
            f.write(f"Max: {coefficients.max():.6f}\n")
