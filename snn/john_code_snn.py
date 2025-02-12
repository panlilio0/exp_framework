"""
6 weights and 2 biases into a single list per SNN
"""

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
        self._weights = []  # a list of weights and a bias (last item in the list)
        self.level = 0.0    # activation level (neuron's internal state - increases based on the weighted inputs and decays over time if no input)
        self.firelog = []   # tracks whether the neuron fired (1) or not (0)
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
            print(f"Error: {len(inputs)} inputs vs {len(self._weights)} weights")
            return 0.0
        # Calculate weighted sum of inputs
        weighted_sum = sum(inputs[i] * self._weights[i] for i in range(len(inputs)))
        self.level = max(self.level + weighted_sum, 0.0)
        print(f"new level: {self.level}")

        # Check if neuron fires
        if self.level >= self.get_bias():
            self.level = 0.0
            self.firelog.append(1)
            return 1.0
        else:
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
        self.nodes = [SpikyNode(num_inputs) for _ in range(num_nodes)]  # list of neurons (SpikyNodes)
    

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
        self.hidden_layer = SpikyLayer(hidden_size, input_size)     # creates a hidden layer with the given parameters
        self.output_layer = SpikyLayer(output_size, hidden_size)    # creates an output layer with the given parameters


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
        # input_weights contains weights for both - hidden and output layers

        # Split weights into two equal parts for hidden and output layers
        half = len(input_weights) // 2

        self.hidden_layer.set_weights(input_weights[:half])
        self.output_layer.set_weights(input_weights[half:])
    

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
    print('\nSpikyNode:')
    node = SpikyNode(5)
    node.print_weights()
    print(node.compute(list(range(1, 6))))
    node.print_weights()
    print('\nSpikyNet:')
    net = SpikyNet(5, 2, 4)
    net.print_structure()
    print('\n')
    print(net.compute(list(range(1, 11, 2))))


# Not sure if this works properly, will need to test if using this for implementing SNN
