import numpy as np
from node import SpikyNode

class SpikyLayer:
    def __init__(self, inp_size, num_nodes):
        self.nodes = [SpikyNode(inp_size) for _ in range(num_nodes)]
    
    def compute(self):
        pass
    
    def size(self):
        return len(self.nodes)