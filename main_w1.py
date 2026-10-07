import torch
from core.utils import create_base_path, get_next_model_id, create_dimensions_string
from core.experiment import run_experiment

if __name__ == "__main__":
    number_of_models = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = "classification"
    n_samples = 15000
    input_neurons = 700
    cluster_std = None
    n_classes = 2
    batch_size = 128
    hidden_layers = 1
    class_seps = [0.5, 5.0]
    cluster_stds = [None]
    architectures = [[100], [150], [200]]
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
