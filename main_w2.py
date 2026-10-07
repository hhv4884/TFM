import torch
from core.experiment import run_experiment


def experiment_1():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = "classification"
    class_sep = 1.0
    n_samples = 30000
    n_classes = 2
    batch_size = 128
    output_neurons = 1
    epochs = 60
    number_of_models = 5
    print("\n")
    print("-." * 20)
    print("1-FIXED CONFIGURATION")
    print("-." * 20)
    architectures = [(1000, [1000])]
    for input_neurons, hidden_neurons in architectures:
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
            )


def experiment_2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = "classification"
    n_samples = 30000
    n_classes = 2
    batch_size = 128
    output_neurons = 1
    epochs = 60
    number_of_models = 5
    print("\n")
    print("-." * 20)
    print("2-DIFFERENT DIFFICULTIES")
    print("-." * 20)
    input_neurons = 1000
    hidden_neurons = [1000]
    class_seps = [
        0.5,
        1.0,
        5.0,
    ]
    for class_sep in class_seps:
        for model_number in range(number_of_models):
            run_experiment(
                task=task,
                n_samples=n_samples,
                input_neurons=input_neurons,
                n_classes=n_classes,
                cluster_std=1.0,
                class_sep=class_sep,
                batch_size=batch_size,
                hidden_layers=1,
                hidden_neurons=hidden_neurons,
                output_neurons=output_neurons,
                device=device,
                thresholds=None,
                epochs=epochs,
                experiment_name="experiment_2",
            )


def experiment_3():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = "classification"
    n_samples = 30000
    n_classes = 2
    batch_size = 128
    output_neurons = 1
    epochs = 60
    number_of_models = 5
    print("\n")
    print("-." * 100)
    print("3-DIFFERENT THRESHOLDS")
    print("-." * 100)
    input_neurons = 1000
    hidden_neurons = [1000]
    thresholds = [None, 0.2, 0.4, 0.6, 0.8]
    for model_number in range(number_of_models):
        run_experiment(
            task=task,
            n_samples=n_samples,
            input_neurons=input_neurons,
            n_classes=n_classes,
            cluster_std=1.0,
            class_sep=1.0,
            batch_size=batch_size,
            hidden_layers=1,
            hidden_neurons=hidden_neurons,
            output_neurons=output_neurons,
            device=device,
            thresholds=thresholds,
            epochs=epochs,
            experiment_name="experiment_3",
        )


def experiment_4():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task = "classification"
    n_samples = 30000
    n_classes = 2
    batch_size = 128
    output_neurons = 1
    epochs = 60
    number_of_models = 5
    print("\n")
    print("-." * 20)
    print("4-DIFFERENT ARCHITECTURE")
    print("-." * 20)
    architectures = [
        (100, [100]),
        (500, [500]),
        (1000, [1000]),
        (10000, [10000]),
    ]
    for input_neurons, hidden_neurons in architectures:
        for model_number in range(number_of_models):
            run_experiment(
                task=task,
                n_samples=n_samples,
                input_neurons=input_neurons,
                n_classes=n_classes,
                cluster_std=1.0,
                class_sep=1.0,
                batch_size=batch_size,
                hidden_layers=1,
                hidden_neurons=hidden_neurons,
                output_neurons=output_neurons,
                device=device,
                thresholds=None,
                epochs=epochs,
                experiment_name="experiment_4",
            )


if __name__ == "__main__":
    experiment_1()
    experiment_2()
    experiment_3()
    experiment_4()
