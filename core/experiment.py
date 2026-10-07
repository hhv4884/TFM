import os
import torch
from .models import MLP
from .datasets import (
    load_dataset,
    save_dataset,
    save_split_index,
    get_dataset_parameter,
    get_next_model_id,
)
from .train import train_model
from .utils import create_base_path, create_dimensions_string
from .graph_analysis import (
    analyze_model_with_thresholds,
    save_final_weight_panels,
    get_best_epoch_from_logs,
)


def run_experiment(
    task,
    n_samples,
    input_neurons,
    n_classes,
    cluster_std,
    class_sep,
    batch_size,
    hidden_layers,
    hidden_neurons,
    output_neurons,
    device,
    epochs,
    thresholds=None,
    experiment_name=None,
):
    dimensions = create_dimensions_string(input_neurons, hidden_neurons, output_neurons)
    parameter_name, parameter_value = get_dataset_parameter(
        task, class_sep, cluster_std
    )
    dataset_base_path = create_base_path(
        "datasets",
        n_samples,
        task,
        dimensions,
        parameter_name,
        parameter_value,
    )
    model_id = get_next_model_id(dataset_base_path)
    dataset_path = os.path.join(dataset_base_path, f"model_{model_id}.csv")
    model_base_path = create_base_path(
        "trained_models",
        n_samples,
        task,
        dimensions,
        parameter_name,
        parameter_value,
        experiment_name=experiment_name
    )
    model_path = os.path.join(model_base_path, f"model_{model_id}")
    os.makedirs(model_path, exist_ok=True)
    train_loader, test_loader, x, y, train_index, test_index = load_dataset(
        task, batch_size, n_samples, input_neurons, n_classes, cluster_std, class_sep
    )
    save_dataset(x, y, dataset_path)
    dataset_split_path = os.path.join(model_path, "dataset_split")
    save_split_index(train_index, test_index, dataset_split_path)
    model = MLP(hidden_layers, input_neurons, hidden_neurons, output_neurons)
    model.to(device)
    train_model(
        model,
        train_loader,
        test_loader,
        model_path,
        device,
        epochs=epochs,
    )
    graph_path = os.path.join(model_path, "graph_analysis")
    logs_path = os.path.join(model_path, "logs")
    checkpoint_path = os.path.join(model_path, "checkpoints")
    best_epoch = get_best_epoch_from_logs(logs_path)
    epochs_to_analyze = sorted(set([0, best_epoch, epochs - 1]))
    for epoch in epochs_to_analyze:
        checkpoint_file = os.path.join(checkpoint_path, f"epoch_{epoch}.pt")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        analyze_model_with_thresholds(
            model=model, graph_path=graph_path, epoch=epoch, thresholds=thresholds
        )
    save_final_weight_panels(
        model_path=model_path, graph_path=graph_path, logs_path=logs_path, epochs=epochs
    )
