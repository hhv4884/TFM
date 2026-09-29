import os
import torch
from core.models import MLP
from core.datasets import (
    load_dataset,
    save_dataset,
    save_split_index,
    get_dataset_parameter,
    get_next_model_id,
)
from core.train import train_model
from core.utils import create_base_path, create_dimensions_string


def run_experiment(
    model_id,
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
):

    print()
    print("-." * 80)
    print(f"MODEL {model_id}")
    print("-." * 80)
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
    dataset_path = os.path.join(dataset_base_path, f"model_{model_id}.csv")
    model_base_path = create_base_path(
        "trained_models",
        n_samples,
        task,
        dimensions,
        parameter_name,
        parameter_value,
    )
    model_path = os.path.join(model_base_path, f"model_{model_id}")
    os.makedirs(model_path, exist_ok=True)
    train_loader, test_loader, x, y, train_index, test_index = load_dataset(
        task,
        batch_size,
        n_samples,
        input_neurons,
        n_classes,
        cluster_std,
        class_sep,
    )
    save_dataset(x, y, dataset_path)
    dataset_split_path = os.path.join(model_path, "dataset_split")
    save_split_index(train_index, test_index, dataset_split_path)
    model = MLP(hidden_layers, input_neurons, hidden_neurons, output_neurons)
    print(model)
    model.to(device)
    train_model(model, train_loader, test_loader, model_path, device)
    print()
    print(f"Model {model_id} finished.")


if __name__ == "__main__":
    number_of_models = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = "classification"
    n_samples = 15000
    input_neurons = 700
    # aunque no lo utilice en esta, para el datasets.py
    cluster_std = 1.0
    n_classes = 2
    batch_size = 128
    hidden_layers = 1
    class_seps = [0.5,5.0]
    architectures = [[100],[150], [200]]
    output_neurons = 1
    print(f"Device: {device}")
    print(f"Task: {task}")
    print(f"Models per configuration: {number_of_models}")

    if task == "classification":
        for class_sep in class_seps:
            for hidden_neurons in architectures:
                hidden_layers = len(hidden_neurons)
                dimensions = create_dimensions_string(
                    input_neurons, hidden_neurons, output_neurons
                )
                parameter_name = "class_sep"
                parameter_value = class_sep
                dataset_base_path = create_base_path(
                    "datasets",
                    n_samples,
                    task,
                    dimensions,
                    parameter_name,
                    parameter_value,
                )
                model_id = get_next_model_id(dataset_base_path)
                print()
                print("-." * 10)
                print("EXPERIMENT CONFIGURATION")
                print("-." * 10)
                print(f"Task:          {task}")
                print(f"Class sep:     {class_sep}")
                print(
                    f"Architecture:  {input_neurons} -> {hidden_neurons} -> {output_neurons}"
                )
                print(f"Models:        {number_of_models}")
                print("-." * 10)

                for i in range(number_of_models):
                    run_experiment(
                        model_id=model_id + i,
                        task=task,
                        n_samples=n_samples,
                        input_neurons=input_neurons,
                        n_classes=n_classes,
                        cluster_std=cluster_std,
                        class_sep=class_sep,
                        batch_size=batch_size,
                        hidden_layers=hidden_layers,
                        hidden_neurons=hidden_neurons,
                        output_neurons=output_neurons,
                        device=device,
                    )
    elif task == "linear":
        for cluster_std in cluster_stds:
            for hidden_neurons in architectures:
                hidden_layers = len(hidden_neurons)
                dimensions = create_dimensions_string(
                    input_neurons, hidden_neurons, output_neurons
                )
                parameter_name = "cluster_std"
                parameter_value = cluster_std
                dataset_base_path = create_base_path(
                    "datasets",
                    n_samples,
                    task,
                    dimensions,
                    parameter_name,
                    parameter_value,
                )
                model_id = get_next_model_id(dataset_base_path)
                print()
                print("-." * 10)
                print("EXPERIMENT CONFIGURATION")
                print("-." * 10)
                print(f"Task:          {task}")
                print(f"Cluster std:   {cluster_std}")
                print(
                    f"Architecture:  {input_neurons} -> {hidden_neurons} -> {output_neurons}"
                )
                print(f"Models:        {number_of_models}")
                print("-." * 10)
                for i in range(number_of_models):
                    run_experiment(
                        model_id=model_id + i,
                        task=task,
                        n_samples=n_samples,
                        input_neurons=input_neurons,
                        n_classes=n_classes,
                        cluster_std=cluster_std,
                        class_sep=5.0,
                        batch_size=batch_size,
                        hidden_layers=hidden_layers,
                        hidden_neurons=hidden_neurons,
                        output_neurons=output_neurons,
                        device=device,
                    )
