import torch
from core.experiment import run_experiment


def experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = "classification"
    n_classes = 2
    batch_size = 128
    epochs = 150
    p_inf = 1 / 10
    p_red = 1 / 4
    alpha = 10
    experiments = [{
        "architectures": [(2500, [1000], n_classes)],
        "difficulties": [1.0],
        "thresholds": [None],
        "n_models": 5},
        {"architectures": [(2500, [1000], n_classes)],
         "difficulties": [0.5, 5.0],
         "thresholds": [None],
         "n_models":1},
        {"architectures": [(2500, [1000], n_classes)],
         "difficulties": [1.0],
         "thresholds": [None, 50, 75, 90, 95, 99],
         "n_models": 1},
        {"architectures": [(1000, [1000], n_classes),
                           (1000, [2500], n_classes),
                           (2500, [2500], n_classes)],
         "difficulties": [1.0],
         "thresholds": [None],
         "n_models":1}]
    for experiment_number, experiment_config in enumerate(experiments, start=1):
        for architecture_number, architecture in enumerate(
            experiment_config["architectures"], start=1
        ):
            input_neurons, hidden_neurons, output_neurons = architecture
            n_samples = input_neurons * alpha
            n_informative = int(p_inf * input_neurons)
            n_redundant = int(p_red * input_neurons)
            for difficulty in experiment_config["difficulties"]:
                for model_number in range(experiment_config["n_models"]):
                    model_id = (
                        f"arch_{architecture_number}"
                        f"_difficulty_{difficulty}"
                        f"_rep_{model_number + 1}"
                    )
                    run_experiment(
                        task=task,
                        n_samples=n_samples,
                        n_classes=output_neurons,
                        n_informative=n_informative,
                        n_redundant=n_redundant,
                        cluster_std=None,
                        class_sep=difficulty,
                        batch_size=batch_size,
                        input_neurons=input_neurons,
                        hidden_layers=len(hidden_neurons),
                        hidden_neurons=hidden_neurons,
                        output_neurons=output_neurons,
                        device=device,
                        thresholds=experiment_config["thresholds"],
                        epochs=epochs,
                        experiment_name=f"experiment_{experiment_number}",
                        model_id=model_id,
                    )


if __name__ == "__main__":
    experiment()
