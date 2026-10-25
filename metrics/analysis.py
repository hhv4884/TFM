import os
import torch
import torch.nn as nn
import numpy as np
from .csv_utils import update_csv_clustering_info, update_csv_degree_info


def degree_distribution(degrees):
    degrees = degrees.detach().cpu().long()
    unique_degrees, counts = torch.unique(degrees, return_counts=True)
    probabilities = counts.float() / len(degrees) * 100
    return unique_degrees, counts, probabilities


def degree_index(weight, epoch, threshold, degree_path, partition_number):
    mask = torch.abs(weight) > threshold
    in_degree = mask.sum(dim=1)
    out_degree = mask.sum(dim=0)
    in_degree_values, in_degree_count, in_degree_probabilities = degree_distribution(
        in_degree
    )
    out_degree_values, out_degree_count, out_degree_probabilities = degree_distribution(
        out_degree
    )
    in_mean = in_degree.float().mean().item()
    in_median = in_degree.float().median().item()
    in_std_dev = in_degree.float().std(unbiased=False).item()
    out_mean = out_degree.float().mean().item()
    out_median = out_degree.float().median().item()
    out_std_dev = out_degree.float().std(unbiased=False).item()
    update_csv_degree_info(
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
    )


# coeficiente de clustering para cada neurona de la primera dimensión de A
def bip_clustering(A):
    A = A.float()
    H = A.shape[0]
    inter = A @ A.T
    deg = A.sum(dim=1)
    union = deg[:, None] + deg[None, :] - inter
    jac = inter / union.clamp(min=1)
    mask = (inter > 0) & ~torch.eye(H, dtype=torch.bool, device=A.device)
    cc = (jac * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return cc.cpu().numpy()


def clustering_index(weight, epoch, threshold, clustering_path, partition_number):
    mask = torch.abs(weight) > threshold
    A = mask.float()
    cc_in = bip_clustering(A)
    mean = float(np.mean(cc_in))
    median = float(np.median(cc_in))
    std_dev = float(np.std(cc_in))
    update_csv_clustering_info(
        epoch, partition_number, "in", cc_in, mean, median, std_dev, clustering_path
    )
    cc_out = bip_clustering(A.T)
    mean = float(np.mean(cc_out))
    median = float(np.median(cc_out))
    std_dev = float(np.std(cc_out))
    update_csv_clustering_info(
        epoch, partition_number, "out", cc_out, mean, median, std_dev, clustering_path
    )


def analyze_partition(weight, epoch, thresholds, csv_path, partition_number):
    weight = weight.detach().cpu()
    for threshold_percentile in thresholds:
        if threshold_percentile is None:
            degree_path = os.path.join(csv_path, "threshold_median")
            clustering_path = os.path.join(csv_path, "threshold_median")
            threshold = torch.median(torch.abs(weight)).item()
        else:
            degree_path = os.path.join(csv_path, f"threshold_{threshold_percentile}")
            clustering_path = os.path.join(csv_path, f"threshold_{threshold_percentile}")
            threshold = torch.quantile(
                torch.abs(weight).float(), threshold_percentile / 100.0
            ).item()
        degree_index(weight, epoch, threshold, degree_path, partition_number)
        clustering_index(weight, epoch, threshold, clustering_path, partition_number)


def analyze_all_partitions(model, epoch, thresholds, csv_path):
    partitions = [layer for layer in model.net if isinstance(layer, nn.Linear)]
    for partition_number, layer in enumerate(partitions, start=1):
        analyze_partition(layer.weight, epoch, thresholds, csv_path, partition_number)
