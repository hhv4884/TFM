import os
import torch
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def load_dataset(
    task,
    batch_size,
    n_samples,
    n_features,
    n_classes,
    cluster_std,
    class_sep,
    n_informative,
    n_redundant,
):
    if task == "linear":
        x, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_classes,
            cluster_std=cluster_std,
        )
    elif task == "classification":
        x, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            class_sep=class_sep,
            n_informative=n_informative,
            n_redundant=n_redundant
        )
    else:
        raise ValueError(f"Invalid task: {task}")

    index = torch.arange(len(x))
    train_index, test_index = train_test_split(
        index.numpy(), test_size=0.2, shuffle=True
    )
    x_train = torch.FloatTensor(x[train_index])
    x_test = torch.Tensor(x[test_index])
    y_train = torch.FloatTensor(y[train_index])
    y_test = torch.Tensor(y[test_index])
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size, True)
    test_loader = DataLoader(test_dataset, batch_size, False)
    return train_loader, test_loader, x, y, train_index, test_index


def save_dataset(x, y, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()
    columns = [f"feature_{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=columns)
    df["target"] = y
    df.to_csv(path, index=False)
    print(f"Dataset saved to: {path}")


def save_split_index(train_index, test_index, path):
    os.makedirs(path, exist_ok=True)
    train_path = os.path.join(path, "train_index.npy")
    test_path = os.path.join(path, "test_index.npy")
    train_index = train_index.astype("int64")
    test_index = test_index.astype("int64")
    np.save(train_path, train_index)
    np.save(test_path, test_index)
    print(f"Train index saved to: {train_path}")
    print(f"Test index saved to: {test_path}")


def get_dataset_parameter(task, class_sep, cluster_std):
    if task == "classification":
        return "class_sep", class_sep
    elif task == "linear":
        return "cluster_std", cluster_std
    else:
        raise ValueError(f"Unknown dataset task: {task}")


def get_next_model_id(dataset_path):
    model_id = 1
    while os.path.exists(os.path.join(dataset_path, f"model_{model_id}.csv")):
        model_id += 1
    return model_id
