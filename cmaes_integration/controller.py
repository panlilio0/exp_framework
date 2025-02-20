"""Module for taking weights from CMAES and inputs from morpho to give an output of target lengths"""

import json
import time
import os
import numpy as np
from snn.john_code_snn import SpikyNet

INPUT_LAYERS = 4
HIDDEN_LAYERS = 2
OUTPUT_SIZE = 1
MAX_LENGTH = 1.6
MIN_LENGTH = 0.6

class Controller:

    def __init__(self, weights):    # weights : nested array ([[weights for SNN 1], [SNN 2], [SNN 3], [SNN 3]])
        self.snns = []
        for i in range(len(weights)):
            new_snn = SpikyNet(INPUT_LAYERS, HIDDEN_LAYERS, OUTPUT_SIZE)
            new_snn.set_weights(weights[i])
            self.snns.append(new_snn)

    def computeAll(self, inputs):   # inputs : nested array ([[inputs for SNN 1], [SNN 2], [SNN 3], [SNN 3]])
        """
        Takes the input for all the SNNs, computes their output, and returns the outputs (target lengths)
        """
        if len(inputs) != len(self.snns):
            print("Mismatched input size")
            return
        else:
            outputs = []
            for i in range(self.snns):
                curr_inputs = inputs[i]
                curr_snn = self.snns[i]
                curr_output = curr_snn.compute(curr_inputs)
                target_length = MAX_LENGTH if curr_output == 1 else MIN_LENGTH
                outputs.append(target_length)
            
            return outputs