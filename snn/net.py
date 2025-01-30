from layers import SpikyLayer
import numpy as np


class SpikyNet:
    def __init__(self, inp_size, hidden_size, out_size):
        self.hidden_layer = SpikyLayer(inp_size, hidden_size)
        self.out_layer = SpikyLayer(hidden_size, out_size)

    def compute(self, x):
        hidden_out = self.hidden_layer.compute(x)
        return self.out_layer.compute(hidden_out)
    
    def __str__(self):
        print(f"Hidden layer: {self.hidden_layer.size()}\nOutput layer: {self.out_layer.size()}")
