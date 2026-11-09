import os
import math
import torch
import torch.nn as nn
import numpy as np
from .csv_utils import (
    update_csv_clustering_info,
    update_csv_degree_info,
    update_csv_strength_info,
    update_csv_weighted_clustering_info,
)


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


def strength(weight):
    abs_weight = torch.abs(weight)
    input_strength = abs_weight.sum(dim=1)
    output_strength = abs_weight.sum(dim=0)
    return input_strength, output_strength


def strength_index(
    incoming_weight, outgoing_weight, epoch, strength_path, hidden_layer
):
    input_strength, _ = strength(incoming_weight)
    _, output_strength = strength(outgoing_weight)
    total_strength = input_strength + output_strength
    n_nodes = (
        incoming_weight.shape[1] + incoming_weight.shape[0] + outgoing_weight.shape[0]
    )
    n_edges = (
        torch.count_nonzero(incoming_weight).item()
        + torch.count_nonzero(outgoing_weight).item()
    )
    update_csv_strength_info(
        epoch,
        hidden_layer,
        input_strength,
        output_strength,
        total_strength,
        strength_path,
        n_edges,
        n_nodes,
    )


def hidden_weighted_bipartite_clustering(fc1, fc2):
    W1 = fc1.weight.detach()
    W2 = fc2.weight.detach()
    K = torch.cat([W1.T, W2], dim=0).double()
    wmax = K.abs().max()
    Kb = K.T
    Ka = Kb.abs()
    Wsq = Ka**2
    P = Kb @ Kb.T
    Q = Wsq @ Wsq.T
    s = Ka.sum(dim=1)
    pair_w = s**2 - Wsq.sum(dim=1)
    num = (P**2).sum(dim=1) - torch.diagonal(P) ** 2 - Q.sum(dim=1) + torch.diagonal(Q)
    n_i = Kb.shape[0] - 1
    c = num / (n_i * wmax**2 * pair_w)
    c[~torch.isfinite(c)] = float("nan")
    return c


def weighted_clustering_index(fc1, fc2, epoch, clustering_path, hidden_layer):
    coefficients = hidden_weighted_bipartite_clustering(fc1, fc2)
    coefficients = coefficients.detach().cpu().numpy()
    mean = float(np.nanmean(coefficients))
    median = float(np.nanmedian(coefficients))
    std_dev = float(np.nanstd(coefficients))
    n_nodes = (fc1.weight.shape[1] + fc1.weight.shape[0] + fc2.weight.shape[0])
    n_edges = (
        torch.count_nonzero(fc1.weight).item() + torch.count_nonzero(fc2.weight).item()
    )
    update_csv_weighted_clustering_info(
        epoch,
        hidden_layer,
        coefficients,
        mean,
        median,
        std_dev,
        clustering_path,
        n_edges,
        n_nodes,
    )


def backbone_index(
    weight,
    epoch,
    alpha,
    backbone_path,
    partition_number,
):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha debe estar entre 0 y 1.")
    abs_weight = torch.abs(weight.detach())
    n_edges = abs_weight.numel()
    n_selected = math.ceil(alpha * n_edges)
    backbone_mask = torch.zeros_like(
        abs_weight,
        dtype=torch.bool,
    )
    if n_selected > 0:
        selected_indices = torch.topk(
            abs_weight.reshape(-1),
            k=n_selected,
            largest=True,
            sorted=False,
        ).indices
        backbone_mask.view(-1)[selected_indices] = True
    backbone_weight = backbone_mask.float()
    degree_index(
        backbone_weight,
        epoch,
        0.5,
        backbone_path,
        partition_number,
    )
    clustering_index(
        backbone_weight,
        epoch,
        0.5,
        backbone_path,
        partition_number,
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
            clustering_path = os.path.join(
                csv_path, f"threshold_{threshold_percentile}"
            )
            threshold = torch.quantile(
                torch.abs(weight).float(), threshold_percentile / 100.0
            ).item()
        degree_index(weight, epoch, threshold, degree_path, partition_number)
        clustering_index(weight, epoch, threshold, clustering_path, partition_number)


def analyze_all_partitions(
    model,
    epoch,
    thresholds,
    csv_path,
    calculate_strength=False,
    calculate_weighted_clustering=False,
    backbone_alpha=0.01,
):
    partitions = [layer for layer in model.net if isinstance(layer, nn.Linear)]
    for partition_number, layer in enumerate(partitions, start=1):
        analyze_partition(layer.weight, epoch, thresholds, csv_path, partition_number)
        if backbone_alpha is not None:
            backbone_path = os.path.join(csv_path, f"backbone_alpha_{backbone_alpha:g}")
            backbone_index(
                layer.weight,
                epoch,
                backbone_alpha,
                backbone_path,
                partition_number,
            )
    if calculate_strength:
        strength_path = os.path.join(csv_path, "weighted_graph", "strength")
        for i in range(len(partitions) - 1):
            strength_index(
                partitions[i].weight,
                partitions[i + 1].weight,
                epoch,
                strength_path,
                hidden_layer=i + 1,
            )

    if calculate_weighted_clustering:
        clustering_path = os.path.join(csv_path, "weighted_graph", "clustering")
        for i in range(len(partitions) - 1):
            weighted_clustering_index(
                partitions[i],
                partitions[i + 1],
                epoch,
                clustering_path,
                hidden_layer=i + 1,
            )
