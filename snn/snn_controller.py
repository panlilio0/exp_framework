"""
Module for running SNN outputs with proper input/output handling.
"""

import json
import os
import numpy as np
from snn.model_struct import SpikyNet

# Constants for SNN configuration
MIN_LENGTH = 0.6  # Minimum actuator length
MAX_LENGTH = 1.6  # Maximum actuator length
_current_file = os.path.abspath(__file__)
_project_root = os.path.dirname(os.path.dirname(_current_file))
ROBOT_DATA_PATH = os.path.join(_project_root, "morpho_demo", "world_data",
                               "bestbot.json")


class SNNController:
    """Class to handle SNN input/output processing."""

    def __init__(self,
                 inp_size,
                 hidden_size,
                 output_size,
                 robot_config=ROBOT_DATA_PATH):
        """Initialize with None - will set sizes after loading robot data."""
        self.snns = []
        self.num_snn = 0  # Number of spiking neural networks (actuators)
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self._load_robot_config(robot_config)

    def _load_robot_config(self, robot_path):
        """
        Load robot configuration from JSON file and initialize SNN.
        
        Args:
            robot_path (str): Path to robot JSON configuration file
            
        Returns:
            tuple: (num_actuators, input_size) - Network dimensions
        """
        if not os.path.exists(robot_path):
            raise FileNotFoundError(
                f"Robot configuration file not found: {robot_path}")
        with open(robot_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Extract robot data
        robot_key = list(data["objects"].keys())[0]
        robot_data = data["objects"][robot_key]
        # Count actuators (types 3 and 4)
        self.num_snn = sum(1 for t in robot_data["types"] if t in [3, 4])
        # Initialize SNN with proper dimensions
        self.snns = [
            SpikyNet(input_size=self.inp_size,
                     hidden_size=self.hidden_size,
                     output_size=self.output_size) for _ in range(self.num_snn)
        ]

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

        params_per_hidden_layer = (self.inp_size + 1) * self.hidden_size
        params_per_output_layer = (self.hidden_size + 1) * self.output_size
        params_per_snn = params_per_hidden_layer + params_per_output_layer

        flat_vector = np.array(cmaes_out)  # np.array(pipeline.get_cmaes_out())

        if flat_vector.size != (self.num_snn * params_per_snn):
            raise ValueError(f"Expected CMA-ES output vector of size \
                             {(self.num_snn * params_per_snn)}, got {flat_vector.size}."
                             )

        # Reshape the flat vector to a 2D array: each row corresponds to one SNN.
        reshaped = flat_vector.reshape((self.num_snn, params_per_snn))

        # For each SNN, split the parameters into weights and biases.
        snn_parameters = {}
        for snn_idx, params_per_snn in enumerate(reshaped):
            hidden_params = params_per_snn[:params_per_hidden_layer]
            output_params = params_per_snn[params_per_hidden_layer:]
            snn_parameters[snn_idx] = {
                'hidden_layer': hidden_params,
                'output_layer': output_params
            }

        for snn_id, params in snn_parameters.items():
            self.snns[snn_id].set_weights(params)

    def _get_output_state(self, inputs):
        """
        Run SNN with inter-actuator distances as input over multiple timesteps.
        
        Args:
            inputs (list): inter-actuator distances
            
        Returns:
            dict: Contains 'continuous_actions' and 'duty_cycles'
        """

        # Normalizing inputs between -1 and 1
        x_vals, y_vals = zip(*inputs)  # Unzips into two lists

        # Find min and max for each component
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)

        # Normalize each component independently
        inputs = [
            (
                2 * (x - x_min) / (x_max - x_min) - 1,  # Normalize x
                2 * (y - y_min) / (y_max - y_min) - 1  # Normalize y
            ) for x, y in inputs
        ]

        outputs = {}
        for snn_id, snn in enumerate(self.snns):
            duty_cycle = snn.compute(inputs[snn_id])
            # Map duty_cycle (assumed in [0,1]) to target length in [MIN_LENGTH, MAX_LENGTH]
            actions = [
                1.6 if duty_cycle[0] > 0.5 else 0.6
            ]
            outputs[snn_id] = {
                "target_length": actions,
                "duty_cycle": duty_cycle
            }

        return outputs

    def get_lengths(self, inputs):
        """
        Returns a list of target lengths (action array)
        """
        out = self._get_output_state(inputs)
        lengths = []
        for _, item in out.items():
            lengths.append(item['target_length'])
        return lengths

    def get_out_layer_firelog(self):
        """
        Return a dictionary with the firelog for each node in the hidden and output
        layers of each SNN in the controller.
        
        Returns:
            dict: Dictionary with structure:
                    {snn_id: {'hidden': [firelog_node_1, firelog_node_2, ...],
                              'output': [firelog_node_1, firelog_node_2, ...]}}
        """
        return {
            i: {
                'hidden': [
                    snn.hidden_layer.nodes[n].firelog
                    for n in range(len(snn.hidden_layer.nodes))
                ],
                'output': [
                    snn.output_layer.nodes[n].firelog
                    for n in range(len(snn.output_layer.nodes))
                ]
            }
            for i, snn in enumerate(self.snns)
        }

    def get_levels_log(self):
        """
        Return a dictionary with the membrane potential levels 
        log for each node in the hidden and output
        layers of each SNN in the controller.
        
        Returns:
            dict: Dictionary with structure:
                    {snn_id: {'hidden': [levels_log_node_1, levels_log_node_2, ...],
                              'output': [levels_log_node_1, levels_log_node_2, ...]}}
        """
        return {
            i: {
                'hidden': [
                    snn.hidden_layer.nodes[n].get_levels_log()
                    for n in range(len(snn.hidden_layer.nodes))
                ],
                'output': [
                    snn.output_layer.nodes[n].get_levels_log()
                    for n in range(len(snn.output_layer.nodes))
                ]
            }
            for i, snn in enumerate(self.snns)
        }
