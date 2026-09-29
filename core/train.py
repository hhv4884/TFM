import os
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import save_logs
from .graph_analysis import save_graph_analysis

# caso 1: aleatoria
# caso 2: compleja
# caso 3: ordenada


def train_model(model, train_loader, test_loader, path, device):
    epochs = 40
    lr = 0.001
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_accuracy_test = 0
    best_epoch = 0
    best_step = 0
    logs_path = path + "/logs"
    weights_path = path + "/weights"
    bias_path = path + "/biases"
    graph_path = path + "/graph_analysis"
    checkpoint_path = path + "/checkpoints"
    os.makedirs(checkpoint_path, exist_ok=True)
    for epoch in range(epochs):
        log_file = save_logs(epoch, logs_path)
        model.train()
        correct_predictions_train_epoch = 0
        total_samples_train_epoch = 0
        for step, (x_batch, y_batch) in enumerate(train_loader):
            correct_predictions_step = 0
            total_samples_step = 0
            x_batch = x_batch.to(device)
            y_batch = y_batch.unsqueeze(1)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            predictions = (outputs > 0.5).float()
            correct_predictions_train_epoch += (predictions == y_batch).sum().item()
            total_samples_train_epoch += x_batch.shape[0]
            epoch_accumulated_train_accuracy = 100 * (
                correct_predictions_train_epoch / total_samples_train_epoch
            )
            correct_predictions_step += (predictions == y_batch).sum().item()
            total_samples_step += x_batch.shape[0]
            train_step_accuracy = 100 * (correct_predictions_step / total_samples_step)
            model.eval()
            correct_predictions_test = 0
            total_samples_test = 0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.unsqueeze(1)
                    y_batch = y_batch.to(device)
                    outputs = model(x_batch)
                    predictions = (outputs > 0.5).float()
                    correct_predictions_test += (predictions == y_batch).sum().item()
                    total_samples_test += x_batch.shape[0]
            test_accuracy_step = 100 * (correct_predictions_test / total_samples_test)
            model.train()
            line_to_add = (
                f"Epoch [{epoch+1}/{epochs}], "
                f"Step [{step+1}/{len(train_loader)}] | "
                f"Loss: {loss.item():.4f} | "
                f"Train Epoch Actual Accurracy (accumulated): {epoch_accumulated_train_accuracy:.2f}% | "
                f"Train Step Accuracy: {train_step_accuracy:.2f}% | "
                f"Test Step Accuracy: {test_accuracy_step:.2f}%"
            )
            print(line_to_add)
            print(line_to_add, file=log_file)
            if test_accuracy_step > best_accuracy_test:
                best_accuracy_test = test_accuracy_step
                best_epoch = epoch
                best_step = step
            if epoch == epochs - 1:
                line_to_add = (
                    f"BEST ACCURACY IN TEST: {best_accuracy_test}\n"
                    f"BEST EPOCH: {best_epoch+1}\n"
                    f"BEST STEP: {best_step+1}\n"
                )
        print(line_to_add, file=log_file)
        log_file.close()
        scheduler.step()
        model.save_edgelist(weights_path, epoch)
        model.save_biaslist(bias_path, epoch)
        save_graph_analysis(model, graph_path, epoch)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        checkpoint_file = os.path.join(checkpoint_path, f"epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_file)
