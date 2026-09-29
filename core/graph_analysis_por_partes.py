import os
import torch
from .models import MLP
from .graph_analysis import save_graph_analysis

def load_model_from_checkpoint(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = MLP(layer_number=checkpoint["layer_number"] - 2,
                input_neurons=checkpoint["input_neurons"],
                hidden_neurons=checkpoint["hidden_neurons"],
                output_neurons=checkpoint["output_neurons"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["epoch"]

def load_checkpoint_epoch(checkpoint_path, epoch, device="cpu"):
    checkpoint_file = os.path.join(checkpoint_path, f"epoch_{epoch}.pt")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: "
                                f"{checkpoint_file}")
    return load_model_from_checkpoint(checkpoint_file, device)

def load_all_checkpoints(checkpoint_path, device="cpu"):
    checkpoint_files = [filename for filename in os.listdir(checkpoint_path) if filename.endswith(".pt")]
    checkpoint_files.sort(key=lambda filename: int(filename.split("_")[1].split(".")[0]))
    models = []
    for filename in checkpoint_files:
        checkpoint_file = os.path.join(checkpoint_path, filename)
        model, epoch = load_model_from_checkpoint(checkpoint_file, device)
        models.append({"epoch": epoch,
                    "model": model,
                    "checkpoint_path": checkpoint_file})
    return models

def analyze_all_checkpoints(checkpoint_path, graph_path, device="cpu"):
    models = load_all_checkpoints(checkpoint_path, device)
    for checkpoint_data in models:
        epoch = checkpoint_data["epoch"]
        model = checkpoint_data["model"]
        print(f"Analyzing checkpoint "
            f"epoch {epoch}")
        save_graph_analysis(model, graph_path, epoch)

def load_weight_from_edgelist(edgelist_path):
    with open(edgelist_path, "r") as f:
        lines = f.readlines()
    header = lines[0].strip()
    parts = header.split("|")
    input_len = int(parts[0].split(":")[1].strip())
    output_len = int(parts[1].split(":")[1].strip())
    weight = torch.zeros(output_len, input_len, dtype=torch.float32)
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        nodes, value = line.split("->")
        value = float(value.strip())
        origin, target = nodes.strip().split()
        origin_index = int(origin.split("_")[1])
        target_index = int(target.split("_")[1])
        weight[target_index, origin_index] = value
    return weight

def load_bias_from_biaslist(biaslist_path):
    with open(biaslist_path, "r") as f:
        lines = f.readlines()
    header = lines[0].strip()
    bias_len = int(header.split(":")[1].strip())
    bias = torch.zeros(bias_len, dtype=torch.float32)
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        neuron, value = line.split("->")
        value = float(value.strip())
        neuron_index = int(neuron.strip().split("_")[1])
        bias[neuron_index] = value
    return bias

def reconstruct_model_from_edgelists(edgelist_paths, biaslist_paths, device="cpu"):
    if len(edgelist_paths) != len(biaslist_paths):
        raise ValueError("Number of edgelists and "
                        "biaslists must match.")
    weights = [load_weight_from_edgelist(path) for path in edgelist_paths]
    biases = [load_bias_from_biaslist(path) for path in biaslist_paths]
    input_neurons = weights[0].shape[1]
    hidden_neurons = [weight.shape[0] for weight in weights[:-1]]
    output_neurons = weights[-1].shape[0]
    layer_number = len(hidden_neurons)
    model = MLP(layer_number=layer_number,
                input_neurons=input_neurons,
                hidden_neurons=hidden_neurons,
                output_neurons=output_neurons)
    linear_layers = [layer for layer in model.net if isinstance(layer, torch.nn.Linear)]
    for layer, weight, bias in zip(linear_layers, weights, biases):
        if layer.weight.shape != weight.shape:
            raise ValueError(f"Weight shape mismatch: "
                            f"{layer.weight.shape} != "
                            f"{weight.shape}")
        if layer.bias.shape != bias.shape:
            raise ValueError(f"Bias shape mismatch: "
                            f"{layer.bias.shape} != "
                            f"{bias.shape}")
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
    model.to(device)
    model.eval()
    return model

def reconstruct_model_from_epoch(weights_path, biases_path, epoch, device="cpu"):
    edgelist_paths = []
    biaslist_paths = []
    layer_number = 0
    while True:
        edgelist_path = os.path.join(weights_path, f"epoch_{epoch}_layer_{layer_number}_edgelist.txt")
        biaslist_path = os.path.join(biases_path, f"epoch_{epoch}_layer_{layer_number}_biaslist.txt")
        if not os.path.exists(edgelist_path):
            break
        if not os.path.exists(biaslist_path):
            raise FileNotFoundError(f"Missing biaslist: "
                                    f"{biaslist_path}")
        edgelist_paths.append(edgelist_path)
        biaslist_paths.append(biaslist_path)
        layer_number += 1
    if not edgelist_paths:
        raise FileNotFoundError(f"No edgelists found for "
                                f"epoch {epoch}")
    return reconstruct_model_from_edgelists(edgelist_paths, biaslist_paths, device)

def analyze_all_reconstructed_epochs(weights_path, biases_path, graph_path, device="cpu"):
    epochs = set()
    for filename in os.listdir(weights_path):
        if not filename.endswith("_edgelist.txt"):
            continue
        parts = filename.split("_")
        epoch = int(parts[1])
        epochs.add(epoch)
    for epoch in sorted(epochs):
        print(f"Analyzing reconstructed "
            f"model from epoch {epoch}")
        model = reconstruct_model_from_epoch(weights_path, biases_path, epoch, device)
        save_graph_analysis(model, graph_path, epoch)
