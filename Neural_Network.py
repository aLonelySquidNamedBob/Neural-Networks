# LIBRARIES
import numpy as np
import turtle

# CLASSES

# Creates a layer in the network
class Layer():

    # Sets up the layer
    def __init__(self, num_nodes_in, num_nodes_out):
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out

        self.weights = [[1 for b in range(num_nodes_in)] for a in range(num_nodes_out)]
        self.biases = [0 for i in range(num_nodes_out)]

    # Returns a string with all relevant information about the layer
    def __repr__(self) -> str:
        value = ""

        for i in range(self.num_nodes_out):
            value += f"node {i + 1}: weights in : {self.weights[i]}, bias : {self.biases[1]} \n"
        value += f"number of nodes : {self.num_nodes_out}"

        return value

    # Calculates the outputs of the layer given some inputs 
    def calculate_outputs(self, inputs):
        outputs = []

        for node_out in range(self.num_nodes_out):
            output = self.biases[node_out]
            for node_in in range(self.num_nodes_in):
                output += inputs[node_in] * self.weights[node_out][node_in]
            outputs.append(output)

        return outputs


class Network():
    def __init__(self, layer_sizes):
        self.num_inputs = layer_sizes[0]
        self.layers = [Layer(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

    def __repr__(self) -> str:
        value = ""
        for i in range(len(self.layers)):
            value += f"Layer {i + 1}: \n{self.layers[i]}\n\n"
        value += f"number of layers : {len(self.layers)}"
        return value

    def draw(self):
        turtle.setup(.75, .50)
        width, height = turtle.window_width(), turtle.window_height()
        t = turtle.Turtle(visible=False)
        t.speed(10)
        t.penup()
        t.right(90)
        margin_x = .05 * width
        margin_y = .05 * height

        # Draw input layer
        t.goto(- width / 2 + margin_x, height / 2 - margin_y)
        for i in range(self.num_inputs):
            t.forward((height - 2 * margin_y) / (self.num_inputs + 1))
            t.dot(10)

        # Draw the hidden and output layers
        for i in range(len(self.layers)):
            t.goto(- width / 2 + margin_x + (i + 1) * (width - 2 * margin_x) / len(self.layers), height / 2 - margin_y)
            for j in range(self.layers[i].num_nodes_out):
                t.forward((height -  2 * margin_y) / (self.layers[i].num_nodes_out + 1))
                t.dot(10)
        
        turtle.mainloop()
        
    def calculate_outputs(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs


network = Network((2, 5, 2))
network.draw()