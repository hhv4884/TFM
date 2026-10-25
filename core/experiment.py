import os
import torch
from .models import MLP
from .datasets import (
    load_dataset,
)
from .train import train_model
from .utils import create_dimensions_string
from graph_analysis.graph_analysis import create_separated_panels


def run_experiment(
    task,
    n_samples,
    n_classes,
    n_informative,
    n_redundant,
    cluster_std,
    class_sep,
    batch_size,
    input_neurons,
    hidden_layers,
    hidden_neurons,
    output_neurons,
    device,
    thresholds,
    epochs,
    experiment_name,
    model_id,
):
    dimensions = create_dimensions_string(input_neurons, hidden_neurons, output_neurons)
    if task == "classification":
        parameter_name = "class_sep"
        parameter_value = class_sep
    elif task == "linear":
        parameter_name = "cluster_std"
        parameter_value = cluster_std
    else:
        raise ValueError("This is not a registered type of task.")
    model_path = os.path.join(
        "trained_models",
        str(n_samples),
        task,
        dimensions,
        parameter_name,
        str(parameter_value),
        experiment_name,
        f"model_{model_id}",
    )
    os.makedirs(model_path, exist_ok=True)
    csv_path = os.path.join(model_path, "csv")
    train_loader, test_loader = load_dataset(
        task,
        batch_size,
        n_samples,
        input_neurons,
        n_classes,
        cluster_std,
        class_sep,
        n_informative,
        n_redundant,
    )
    model = MLP(
        hidden_layers,
        input_neurons,
        hidden_neurons,
        output_neurons,
    )
    model.to(device)
    train_model(
        model,
        train_loader,
        test_loader,
        csv_path,
        device,
        epochs,
        thresholds,
    )
    create_separated_panels(
        model_path=model_path,
        graph_path=csv_path,
        logs_path=None,
        epochs=epochs,
    )
    return model_path
