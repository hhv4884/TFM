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
