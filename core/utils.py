import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.datasets import make_blobs, make_classification


def create_base_path(
    root,
    n_samples,
    task,
    dimensions,
    parameter_name,
    parameter_value,
    experiment_name=None,
):
    if experiment_name is not None:
        path = os.path.join(
            root,
            experiment_name,
            f"n_samples_{n_samples}",
            task,
            dimensions,
            f"{parameter_name}_{parameter_value}",
        )
    else:
        path = os.path.join(
            root,
            f"n_samples_{n_samples}",
            task,
            dimensions,
            f"{parameter_name}_{parameter_value}",
        )
    os.makedirs(path, exist_ok=True)
    return path

def create_dimensions_string(input_neurons, hidden_neurons, output_neurons):
    dimensions = [input_neurons]
    for i in hidden_neurons:
        dimensions.append(i)
    dimensions.append(output_neurons)
    return "dim_" + "_".join(map(str, dimensions))


def save_logs(epoch, path):
    os.makedirs(path, exist_ok=True)
    log_file = f"{path}/log_epoch_{epoch}.txt"
    log_file = open(log_file, "w")
    return log_file


def paint_dataset_sample(task):
    if task == "linear":
        x, y = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=5.0)
    else:
        x, y = make_classification(1000, 2, 0.5, 2, 0)
    # print(x.shape)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.xlabel("feature 0")
    plt.ylabel("feature 1")
    plt.title(f"dataset: {task}")
    print(os.getcwd())
    plt.savefig("dataset_images/dataset_blobs.png", dpi=300, bbox_inches="tight")
    plt.show()
