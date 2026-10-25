import os


def create_dimensions_string(input_neurons, hidden_neurons, output_neurons):
    dimensions = [input_neurons]
    for i in hidden_neurons:
        dimensions.append(i)
    dimensions.append(output_neurons)
    return "dim_" + "_".join(map(str, dimensions))
