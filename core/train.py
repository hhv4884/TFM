import torch
import torch.nn as nn
import torch.optim as optim
from metrics.csv_utils import update_csv_epoch_info


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).long().view(-1)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        batch_size = y_batch.size(0)
        predictions = outputs.argmax(dim=1)
        correct = (predictions == y_batch).sum().item()
        total_loss += loss.item() * batch_size
        total_correct += correct
        total_samples += batch_size
    train_loss = total_loss / total_samples
    train_accuracy = 100.0 * total_correct / total_samples
    return train_loss, train_accuracy


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    n_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).long().view(-1)
            outputs = model(x)
            loss = criterion(outputs, y)
            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            correct += (outputs.argmax(dim=1) == y).sum().item()
            n_samples += batch_size
    return (total_loss / n_samples, 100.0 * (correct / n_samples))


def train_model(model, train_loader, test_loader, csv_path, device, epochs, thresholds):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    epoch_0_train_loss, epoch_0_train_accuracy = evaluate_model(
        model, train_loader, criterion, device
    )
    epoch_0_test_loss, epoch_0_test_accuracy = evaluate_model(
        model, test_loader, criterion, device
    )
    update_csv_epoch_info(
        model,
        0,
        epoch_0_train_loss,
        epoch_0_train_accuracy,
        epoch_0_test_loss,
        epoch_0_test_accuracy,
        csv_path,
        thresholds,
    )
    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        update_csv_epoch_info(
            model,
            epoch + 1,
            train_loss,
            train_accuracy,
            test_loss,
            test_accuracy,
            csv_path,
            thresholds,
        )
        scheduler.step()
    return csv_path
