import torch


def degree_distribution(degrees):
    degrees = degrees.detach().cpu().long()
    unique_degrees, counts = torch.unique(degrees, return_counts=True)
    probabilities = counts.float() / len(degrees) * 100
    return unique_degrees, counts, probabilities


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


def analyze_partition(weight, threshold=None, threshold_mode="absolute"):
    weight = weight.detach().cpu()
    if threshold is None:
        threshold = torch.median(torch.abs(weight)).item()
    elif threshold_mode=="percentile":
        threshold = torch.quantile(torch.abs(weight).float(),threshold/100.0).item()
    mask = torch.abs(weight) > threshold
    num_connections = mask.sum().item()
    min_weight = weight.min().item()
    max_weight = weight.max().item()

    in_degree = mask.sum(dim=1)
    out_degree = mask.sum(dim=0)

    in_degree_values, in_degree_count, in_degree_probabilities = degree_distribution(
        in_degree
    )
    out_degree_values, out_degree_count, out_degree_probabilities = degree_distribution(
        out_degree
    )

    return {
        "threshold": threshold,
        "mask": mask,
        "num_connections": num_connections,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "in_degree_values": in_degree_values,
        "in_degree_counts": in_degree_count,
        "in_degree_probabilities": in_degree_probabilities,
        "out_degree_values": out_degree_values,
        "out_degree_counts": out_degree_count,
        "out_degree_probabilities": out_degree_probabilities,
    }
