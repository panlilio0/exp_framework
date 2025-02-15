import random

SPIKE_DECAY = 0.1
MAX_BIAS = 1
MAX_FIRELOG_SIZE = 200
FIRELOG_THRESHOLD = 30
"""
Simulates a spiky neuron
"""


class SpikyNode:

    def __init__(self, size):
        self._weights = [
        ]  # a list of weights and a bias (last item in the list)
        self.level = 0.0  # activation level (neuron's internal state - increases based on the weighted inputs and decays over time if no input)
        self.firelog = []  # tracks whether the neuron fired (1) or not (0)
        self.init(size)

    def init(self, size):
        self.firelog.clear()
        if size > 0:
            self._weights = []
            # Initialize input weights between -1 and 1
            for _ in range(size):
                self._weights.append((random.uniform(0, 1) * 2) - 1)
            # Add bias weight between 0 and MAX_BIAS
            self._weights.append(random.uniform(0, 1) * MAX_BIAS)

    """
    decays activation
    computes the weighted sum of inputs
    fires if the activation exceeds the bias, otherwise, continues accumulating
    """

    def compute(self, inputs):
        # Maintain firelog size (max 200 entries)
        while len(self.firelog) > MAX_FIRELOG_SIZE:
            self.firelog.pop(0)

        # Decay the neuron’s activation level
        print(f"current level: {self.level}, bias: {self.get_bias()}")
        self.level = max(self.level - SPIKE_DECAY, 0.0)

        # Validate input dimensions
        if (len(inputs) + 1) != len(self._weights):
            print(
                f"Error: {len(inputs)} inputs vs {len(self._weights)} weights")
            return 0.0
        # Calculate weighted sum of inputs
        weighted_sum = sum(inputs[i] * self._weights[i]
                           for i in range(len(inputs)))
        self.level = max(self.level + weighted_sum, 0.0)
        print(f"new level: {self.level}")

        # Check if neuron fires
        if self.level >= self.get_bias():
            print("Fired --> activation level reset to 0.0\n")
            self.level = 0.0
            self.firelog.append(1)
            return 1.0
        else:
            print("\n")
            self.firelog.append(0)
            return 0.0

    """
    measures how frequently the neuron fires
    """

    def duty_cycle(self):
        if len(self.firelog) == 0:
            return 0.0
        fires = sum(self.firelog)
        if len(self.firelog) > FIRELOG_THRESHOLD:
            return fires / len(self.firelog)
        return 0.0

    """
    sets a weight for a particular node
    """

    def set_weight(self, idx, val):
        if 0 <= idx < len(self._weights):
            self._weights[idx] = val
        else:
            print(f"Invalid weight index: {idx}")

    """
    allows to set the nueron's weights
    """

    def set_weights(self, input_weights):
        if len(input_weights) != len(self._weights):
            print("Weight size mismatch in node")
        else:
            self._weights = input_weights.copy()

    """
    sets the nueron's bias
    """

    def set_bias(self, val):
        self._weights[-1] = val

    """
    returns the bias from the combined list of weights and bias
    bias is the last item in the list
    """

    def get_bias(self):
        return self._weights[-1]

    """
    prints the combined list of weights and bias
    """

    def print_weights(self):
        print(self._weights)


"""
Collection of mutltiple neurons (SpikyNodes)
"""


class SpikyLayer:

    def __init__(self, num_nodes, num_inputs):
        self.nodes = [SpikyNode(num_inputs) for _ in range(num_nodes)
                      ]  # list of neurons (SpikyNodes)

    """
    feeds input to each node and returns their output
    """

    def compute(self, inputs):
        return [node.compute(inputs) for node in self.nodes]

    """
    sets weights for all the neurons in the layer
    """

    def set_weights(self, input_weights):
        if not self.nodes:
            return
        weights_per_node = len(input_weights) // len(self.nodes)
        for i, node in enumerate(self.nodes):
            start = i * weights_per_node
            end = start + weights_per_node
            node.set_weights(input_weights[start:end])

    """
    returns the duty cycles for the neurons in the layer
    """

    def duty_cycles(self):
        return [node.duty_cycle() for node in self.nodes]


"""
Combines 2 spiky layers
"""


class SpikyNet:

    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = SpikyLayer(
            hidden_size,
            input_size)  # creates a hidden layer with the given parameters
        self.output_layer = SpikyLayer(
            output_size,
            hidden_size)  # creates an output layer with the given parameters

    """
    passes the input through the hidden layer
    uses the hidden layer's output as input for the output layer
    """

    def compute(self, inputs):
        hidden_output = self.hidden_layer.compute(inputs)
        return self.output_layer.compute(hidden_output)

    """
    assigns weights to the hidden and the output layer
    """

    def set_weights(self, input_weights):
        # get the total number of weights in the hidden layer
        weights_per_hidden_node = len(self.hidden_layer.nodes[0]._weights)
        total_hidden_node_weights = len(self.hidden_layer.nodes) * weights_per_hidden_node
        
        # get the total number of weights in the output layer
        weights_per_output_node = len(self.output_layer.nodes[0]._weights)
        total_output_node_weights = len(self.output_layer.nodes) * weights_per_output_node
        
        # check if the input array has the correct number of weights (hidden layer + output layer)
        if len(input_weights) == total_hidden_node_weights + total_output_node_weights:
            self.hidden_layer.set_weights(input_weights[:total_hidden_node_weights])
            self.output_layer.set_weights(input_weights[total_hidden_node_weights:])
        else:
            print("Total weight list size mismatch!")    
    

    """
    displays the network weights
    """

    def print_structure(self):
        print("\nHidden Layer:")
        for i, node in enumerate(self.hidden_layer.nodes):
            print(f"Node {i}: ", end="")
            node.print_weights()
        print("\nOutput Layer:")
        for i, node in enumerate(self.output_layer.nodes):
            print(f"Node {i}: ", end="")
            node.print_weights()


# testing

if __name__=='__main__':

    # ----- Test a single SpikyNode -----

    print("\n--- Testing SpikyNode ---")
    test_node = SpikyNode(5)
    print("Initial weights:")
    test_node.print_weights()

    # setting weights manually
    print("\nSetting weights manually")
    test_weights = [0.7, -0.4, 0.9, 0.0, -0.2, 0.8]
    test_node.set_weights(test_weights)
    print("Updated weights:")
    test_node.print_weights()

    print("\nGetting '1' output for manual input")
    output = test_node.compute([1, 2, 3, 4, 5])
    print("Output:", output)

    print("\nGetting '0' output for manual input")
    test_weights = [0.7, -0.4, -0.9, 0.0, -0.2, 0.8]
    test_node.set_weights(test_weights)
    print("Updated weights:")
    test_node.print_weights()
    output = test_node.compute([1, 2, 3, 4, 5])
    print("Output:", output)
    

    # ----- Testing a SpikyLayer (3 nodes, 4 inputs) -----

    print("\n--- Testing SpikyLayer ---")
    test_layer = SpikyLayer(3, 4)
    test_inputs = [1, 2, 3, 4]
    layer_outputs = test_layer.compute(test_inputs)
    print("SpikyLayer outputs:", layer_outputs)

    # setting layer weights manually
    # each node requires 4 weights + 1 bias = 5 weights --> for 3 nodes, total 15 items
    print("\nSetting weights manually")
    test_layer_weights = [0.1 * i for i in range(15)]
    test_layer.set_weights(test_layer_weights)
    print("Updated weights:")
    for i, n in enumerate(test_layer.nodes):
        print(f"Node {i} weights:", n._weights)
    

    # ----- Testing SpikyNet (2 hidden nodes, 3 output nodes, 4 inputs) -----

    print("\n--- Testing SpikyNet ---")
    test_net = SpikyNet(4, 2, 3)
    print("Original structure:")
    test_net.print_structure()
    
    print("\nTesting computing")
    test_net_output = test_net.compute([1, 2, 3, 4])
    print("SpikyNet output:", test_net_output)
    
    # Setting weights manually for the network
    print("\nSetting weights manually")
    test_net_weights = [0.1, 0.2, 0.3, 0.4, 1.0, 
                        0.5, 0.6, 0.7, 0.8, 0.9,
                        1.0, 1.0, 0.7,               
                        0.0, 0.0, 0.5,                
                        0.5, 0.5, 0.8]
    test_net.set_weights(test_net_weights)
    print("Updated weights:")
    test_net.print_structure()
    
    print("Getting output for the updated weights")
    test_net_output = test_net.compute([1, 2, 3, 4])
    print("SpikyNet output:", test_net_output)