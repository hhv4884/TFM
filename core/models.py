import os
import torch
import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, layer_number, input_neurons, hidden_neurons, output_neurons):
        super(MLP, self).__init__()
        capas_dict = OrderedDict()
        if layer_number > 1:
            capas_dict["input_layer"] = nn.Linear(input_neurons, hidden_neurons[0])
            capas_dict["activation_layer_0"] = nn.ReLU()
            for i in range(0, layer_number - 1):
                capas_dict[f"hidden_layer_{i+1}"] = nn.Linear(
                    hidden_neurons[i], hidden_neurons[i + 1]
                )
                capas_dict[f"activation_layer_{i+1}"] = nn.ReLU()
            capas_dict["output_layer"] = nn.Linear(hidden_neurons[-1], output_neurons)
            capas_dict[f"activation_layer_{layer_number}"] = nn.Sigmoid()
        else:
            capas_dict["input_layer"] = nn.Linear(input_neurons, hidden_neurons[0])
            capas_dict["activation_layer_1"] = nn.ReLU()
            capas_dict["output_layer"] = nn.Linear(hidden_neurons[0], 1)
            capas_dict["activation_layer_2"] = nn.Sigmoid()
        self.net = nn.Sequential(capas_dict)
        self.layer_number = layer_number + 2
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

    def forward(self, x):
        output = self.net(x)
        return output

    def extract_weights(self, tensor, origin_layer):
        edgelist = {}
        input_len = tensor.shape[1]
        output_len = tensor.shape[0]
        for x in range(output_len):
            for y in range(input_len):
                input_output_text = f"L{origin_layer}_{y} L{origin_layer+1}_{x}"
                edgelist[input_output_text] = tensor[x][y].item()
        return edgelist, input_len, output_len

    def extract_bias(self, tensor, origin_layer):
        biaslist = {}
        bias_len = tensor.shape[0]
        for i in range(bias_len):
            bias_text = f"L{origin_layer}_{i}"
            biaslist[bias_text] = tensor[i].item()
        return biaslist, bias_len

    def save_edgelist(self, path, epoch):
        os.makedirs(path, exist_ok=True)
        layer_number = 0
        for name, layer in self.net.named_children():
            if isinstance(layer, nn.Linear):
                edgelist, input_len, output_len = self.extract_weights(
                    layer.weight, layer_number
                )
                weight_file = f"{path}/epoch_{epoch}_layer_{layer_number}_edgelist.txt"
                with open(weight_file, "w") as f:
                    f.write(
                        f"Number of origin neurons: {input_len} | "
                        f"Number of target neurons: {output_len}\n"
                    )
                    for key, value in edgelist.items():
                        f.write(f"{key} -> {value}\n")
                layer_number += 1

    def save_biaslist(self, path, epoch):
        os.makedirs(path, exist_ok=True)
        layer_number = 0
        for name, layer in self.net.named_children():
            if isinstance(layer, nn.Linear):
                biaslist, size = self.extract_bias(layer.bias, layer_number)
                bias_file = f"{path}/epoch_{epoch}_layer_{layer_number}_biaslist.txt"
                with open(bias_file, "w") as f:
                    f.write(f"Number of biases: {size}\n")
                    for key, value in biaslist.items():
                        f.write(f"{key} -> {value}\n")
                layer_number += 1
