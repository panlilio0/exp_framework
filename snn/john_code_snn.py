import random


SPIKE_DECAY = 0.1
MAX_BIAS = 10


class SpikyNode:
    def __init__(self, size):
        self._weights = []
        self.level = 0.0
        self.firelog = []
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
    
    def compute(self, inputs):
        # Maintain firelog size (max 200 entries)
        while len(self.firelog) > 200:
            self.firelog.pop(0)
        # Decay the neuron’s activation level
        self.level = max(self.level - SPIKE_DECAY, 0.0)
        # Validate input dimensions
        if (len(inputs) + 1) != len(self._weights):
            print(f"Error: {len(inputs)} inputs vs {len(self._weights)-1} weights")
            return 0.0
        # Calculate weighted sum of inputs
        weighted_sum = sum(inputs[i] * self._weights[i] for i in range(len(inputs)))
        self.level += weighted_sum
        # Check if neuron fires
        if self.level >= self.bias():
            self.level = 0.0
            self.firelog.append(1)
            return 1.0
        else:
            self.firelog.append(0)
            return 0.0
    
    def duty_cycle(self):
        if len(self.firelog) == 0:
            return 0.0
        fires = sum(self.firelog)
        if len(self.firelog) > 30:
            return fires / len(self.firelog)
        return 0.0
    
    def set_weights(self, inws):
        if len(inws) != len(self._weights):
            print("Weight size mismatch in node")
        else:
            self._weights = inws.copy()
    
    def bias(self):
        return self._weights[-1]
    
    def set_bias(self, val):
        self._weights[-1] = val
    
    def set_weight(self, idx, val):
        if 0 <= idx < len(self._weights):
            self._weights[idx] = val
        else:
            print(f"Invalid weight index: {idx}")
    
    def print_weights(self):
        print(self._weights)


class SpikyLayer:
    def __init__(self, num_nodes, num_inputs):
        self.nodes = []
        self.init(num_nodes, num_inputs)
    
    def init(self, num_nodes, num_inputs):
        self.nodes = [SpikyNode(num_inputs) for _ in range(num_nodes)]
    
    def compute(self, inputs):
        return [node.compute(inputs) for node in self.nodes]
    
    def set_weights(self, inws):
        if not self.nodes:
            return
        weights_per_node = len(inws) // len(self.nodes)
        for i, node in enumerate(self.nodes):
            start = i * weights_per_node
            end = start + weights_per_node
            node.set_weights(inws[start:end])
    
    def duty_cycles(self):
        return [node.duty_cycle() for node in self.nodes]


class SpikyNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = SpikyLayer(hidden_size, input_size)
        self.output_layer = SpikyLayer(output_size, hidden_size)

    def compute(self, inputs):
        hidden_output = self.hidden_layer.compute(inputs)
        # Return duty cycles of output layer instead of raw spikes
        return self.output_layer.compute(hidden_output)
    
    def set_weights(self, inws):
        # Split weights into two equal parts for hidden and output layers
        half = len(inws) // 2
        self.hidden_layer.set_weights(inws[:half])
        self.output_layer.set_weights(inws[half:])
    
    def print_structure(self):
        print("Hidden Layer:")
        for i, node in enumerate(self.hidden_layer.nodes):
            print(f"Node {i}: ", end="")
            node.print_weights()
        print("\nOutput Layer:")
        for i, node in enumerate(self.output_layer.nodes):
            print(f"Node {i}: ", end="")
            node.print_weights()


# Not sure if this works porperly, will need to test if using this for implementing SNN