import torch
from core.experiment import run_experiment


def experiment_1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = "classification"
    class_sep = 1.0
    alpha = 10
    n_classes = 2
    batch_size = 128
    output_neurons = n_classes
    epochs = 150
    number_of_models = 5
    print("\n")
    print("-." * 10)
    print("1-FIXED CONFIGURATION")
    print("-." * 10)
    architectures = [(1000, [1000]), (1000, [2500]), (2500, [1000]), (2500, [2500])]
    p_inf = 1 / 10
    p_red = 1 / 4
    for input_neurons, hidden_neurons in architectures:
        n_samples = alpha * input_neurons
        n_informative = int(p_inf * input_neurons)
        n_redundant = int(p_red * input_neurons)
        for model_number in range(number_of_models):
            run_experiment(
                task=task,
                n_samples=n_samples,
                input_neurons=input_neurons,
                n_classes=n_classes,
                cluster_std=None,
                class_sep=class_sep,
                batch_size=batch_size,
                hidden_layers=len(hidden_neurons),
                hidden_neurons=hidden_neurons,
                output_neurons=output_neurons,
                device=device,
                thresholds=None,
                epochs=epochs,
                experiment_name="experiment_1",
                n_informative=n_informative,
                n_redundant=n_redundant,
            )


def experiment_2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = "classification"
    alpha = 10
    n_classes = 2
    batch_size = 128
    output_neurons = n_classes
    epochs = 150
    number_of_models = 1
    print("\n")
    print("-." * 10)
    print("2-DIFFERENT DIFFICULTIES AND ARCHITECTURES")
    print("-." * 10)
    architectures = [
        (2500, [1000]),
    ]
    class_seps = [0.5, 5.0]
    p_inf = 1 / 10
    p_red = 1 / 4
    for input_neurons, hidden_neurons in architectures:
        n_samples = alpha * input_neurons
        n_informative = int(p_inf * input_neurons)
        n_redundant = int(p_red * input_neurons)
        for class_sep in class_seps:
            for model_number in range(number_of_models):
                run_experiment(
                    task=task,
                    n_samples=n_samples,
                    input_neurons=input_neurons,
                    n_classes=n_classes,
                    cluster_std=None,
                    class_sep=class_sep,
                    batch_size=batch_size,
                    hidden_layers=len(hidden_neurons),
                    hidden_neurons=hidden_neurons,
                    output_neurons=output_neurons,
                    device=device,
                    thresholds=None,
                    epochs=epochs,
                    experiment_name="experiment_2",
                    n_informative=n_informative,
                    n_redundant=n_redundant,
                )


def experiment_3():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = "classification"
    alpha = 10
    n_classes = 2
    batch_size = 128
    output_neurons = n_classes
    epochs = 150
    number_of_models = 1
    print("\n")
    print("-." * 10)
    print("3-DIFFERENT THRESHOLDS AND ARCHITECTURES")
    print("-." * 10)
    architectures = [
        (2500, [1000]),
    ]
    p_inf = 1 / 10
    p_red = 1 / 4
    thresholds = [None, 75, 90, 95, 99]
    for input_neurons, hidden_neurons in architectures:
        n_samples = alpha * input_neurons
        n_informative = int(p_inf * input_neurons)
        n_redundant = int(p_red * input_neurons)
        for model_number in range(number_of_models):
            run_experiment(
                task=task,
                n_samples=n_samples,
                input_neurons=input_neurons,
                n_classes=n_classes,
                cluster_std=None,
                class_sep=1.0,
                batch_size=batch_size,
                hidden_layers=len(hidden_neurons),
                hidden_neurons=hidden_neurons,
                output_neurons=output_neurons,
                device=device,
                thresholds=thresholds,
                epochs=epochs,
                experiment_name="experiment_3",
                n_informative=n_informative,
                n_redundant=n_redundant,
            )


if __name__ == "__main__":
    experiment_3()
