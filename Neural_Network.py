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
        t = self.setup_turtle()
        width, height = turtle.window_width(), turtle.window_height()
        margin_x = .05 * width
        margin_y = .05 * height

        # Calculate the position of each node
        points = self.calculate_node_positions(width, height, margin_x, margin_y)

        # Draw the nodes
        self.draw_nodes(t, points)

        # Draw the connections between layers 
        self.draw_connections(t, points)
        
        turtle.mainloop()

    def setup_turtle(self):
        turtle.setup(.75, .50)
        t = turtle.Turtle(visible=False)
        t.speed(10)
        t.penup()
        t.right(90)
        return t

    def calculate_node_positions(self, width, height, margin_x, margin_y):
        # Calculate the position of each node on the input layer
        points = [[[- width / 2 + margin_x,
                    (point + 1) * (height - 2 * margin_y) / (self.num_inputs + 1) - (height / 2) + margin_y] 
                    for point in range(self.num_inputs)]]

        # Calculate the position of eath node for each hidden and output layer  
        for layer in range(len(self.layers)):
            points.append([])
            for point in range(self.layers[layer].num_nodes_out):
                points[layer + 1].append([])
                points[layer + 1][point].append(layer * (width - 2 * margin_x) / (len(self.layers)))
                points[layer + 1][point].append((point + 1) * (height - 2 * margin_y) / (self.layers[layer].num_nodes_out + 1) - height / 2 + margin_y)
        
        return points

    def draw_nodes(self, t, points):
        for layer in points:
            for point in layer:
                t.goto(point)
                t.dot(10)

    def draw_connections(self, t, points):
        for layer_index in range(1, len(points)):
            for point_index in range(len(points[layer_index])):
                for point_in_previous_layer_index in range(len(points[layer_index - 1])):
                    t.goto(points[layer_index][point_index])
                    t.pendown()
                    t.pensize(round(self.layers[layer_index - 1].weights[point_index][point_in_previous_layer_index]))
                    t.goto(points[layer_index - 1][point_in_previous_layer_index])
                    t.penup()
        
    def calculate_outputs(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs


network = Network((10, 5, 2))
network.draw()