import torch
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
            n_redundant=n_redundant,
        )
    else:
        raise ValueError("This is not a registered type of task.")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, shuffle=True, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train = torch.FloatTensor(x_train)
    x_test = torch.Tensor(x_test)
    # crossentropyloss() espera torch.long {0, 1}
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size, True)
    test_loader = DataLoader(test_dataset, batch_size, False)
    return train_loader, test_loader
