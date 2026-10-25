import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, layer_number, input_neurons, hidden_neurons, output_neurons):
        super(MLP, self).__init__()
        if layer_number < 1:
            raise ValueError("layer_number debe ser >= 1")
        if len(hidden_neurons) != layer_number:
            raise ValueError(
                "hidden_neurons debe tener una entrada por cada capa oculta"
            )
        if output_neurons < 2:
            raise ValueError("output_neurons debe ser >= 2")
        capas_dict = OrderedDict()
        capas_dict["input_layer"] = nn.Linear(input_neurons, hidden_neurons[0])
        capas_dict["activation_layer_0"] = nn.ReLU()
        for i in range(1, layer_number):
            capas_dict[f"hidden_layer_{i}"] = nn.Linear(
                hidden_neurons[i - 1], hidden_neurons[i]
            )
            capas_dict[f"activation_layer_{i}"] = nn.ReLU()
        capas_dict["output_layer"] = nn.Linear(hidden_neurons[-1], output_neurons)
        self.net = nn.Sequential(capas_dict)
        self.layer_number = layer_number + 2
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        nn.init.kaiming_normal_(self.net.input_layer.weight, nonlinearity="relu")
        nn.init.zeros_(self.net.input_layer.bias)
        nn.init.xavier_normal_(self.net.output_layer.weight)
        nn.init.zeros_(self.net.output_layer.bias)

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
