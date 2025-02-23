"""
Module for running SNN outputs with proper input/output handling.
"""

import json
import os
import numpy as np
from john_code_snn import SpikyNet

# Constants for SNN configuration
MIN_LENGTH = 0.6  # Minimum actuator length
MAX_LENGTH = 1.6  # Maximum actuator length


class SNNRunner:
    """Class to handle SNN input/output processing."""
    def __init__(self, num_actuators):
        """Initialize with None - will set sizes after loading robot data."""
        self.snns = []
        self.grid_width = None
        self.grid_height = None
        self.indices = None
        self.types = None
        self.neighbors = None
        self.NUM_SNN = num_actuators  # Number of spiking neural networks (actuators)
        self.update_constants()
    
    def update_constants(self):
        self.INP_SIZE = self.NUM_SNN - 1  # Distances to every other actuator
        self.HIDDEN_SIZE = 3
        self.OUTPUT_SIZE = 1
        self.PARAMS_PER_HIDDEN_LAYER = (self.INP_SIZE + 1) * self.HIDDEN_SIZE
        self.PARAMS_PER_OUTPUT_LAYER = (self.HIDDEN_SIZE + 1) * self.OUTPUT_SIZE
        self.PARAMS_PER_SNN = self.PARAMS_PER_HIDDEN_LAYER + self.PARAMS_PER_OUTPUT_LAYER
        
    def set_snn_weights(self, cmaes_out):
        """
        Retrieve the flat CMA-ES output and 
        reshape it into a structured format for the SNN's `set_weights()`.

        Returns:
            snn_parameters: A dictionary containing the weights and biases for each SNN.
                        - dict with two elements : 'hidden_layer' and 'output_layer'
                            'hidden_layer' - weights and biases for all nodes in the hidden layer
                            'output_layer' - weights and biases for all nodes in the output layer
        Raises:
            ValueError: If the length of the CMA-ES output does not match the expected size.
        """
        
        flat_vector = np.array(cmaes_out)  # np.array(pipeline.get_cmaes_out())

        if flat_vector.size != (self.NUM_SNN * self.PARAMS_PER_SNN):
            raise ValueError(f"Expected CMA-ES output vector of size {(self.NUM_SNN * self.PARAMS_PER_SNN)}, got {flat_vector.size}.")

        # Reshape the flat vector to a 2D array: each row corresponds to one SNN.
        reshaped = flat_vector.reshape((self.NUM_SNN, self.PARAMS_PER_SNN))

        # For each SNN, split the parameters into weights and biases.
        snn_parameters = {}
        for snn_idx, params_per_snn in enumerate(reshaped):
            hidden_params = params_per_snn[:self.PARAMS_PER_HIDDEN_LAYER]
            output_params = params_per_snn[self.PARAMS_PER_HIDDEN_LAYER:]
            snn_parameters[snn_idx] = {
                'hidden_layer': hidden_params,
                'output_layer': output_params
            }
        
        for snn_id, params in snn_parameters.items():
            # print(f"SNN {snn_id} params: {params}")
            self.snns[snn_id].set_weights(params)
            print(f"SNN {snn_id}:")
            self.snns[snn_id].print_structure()
    
    def get_output_state(self, inputs):
        """
        Run SNN with inter-actuator distances as input over multiple timesteps.
        
        Args:
            inputs (list): inter-actuator distances
            
        Returns:
            dict: Contains 'continuous_actions' and 'duty_cycles'
        """
        # inputs = self.calculate_corner_distances()
        outputs = {}
        for snn_id, snn in enumerate(self.snns):
            snn.compute(inputs[snn_id])
            duty_cycle = snn.output_layer.duty_cycles()
            scale_factor = MAX_LENGTH - MIN_LENGTH
            scaled_actions = [(dc * scale_factor) + MIN_LENGTH for dc in duty_cycle]
            outputs[snn_id] = {
                "target_length": scaled_actions,
                "duty_cycle": duty_cycle
                }
        return outputs
