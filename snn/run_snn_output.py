"""
Module for running SNN outputs with proper input/output handling.
"""

import json
import time
import os
import numpy as np
from john_code_snn import SpikyNet

# Constants for SNN configuration
MIN_LENGTH = 0.6  # Minimum actuator length
MAX_LENGTH = 1.6  # Maximum actuator length
ROBOT_DATA_PATH = os.path.join("..", "morpho_demo", "world_data", "bestbot.json")


class SNNRunner:
    """Class to handle SNN input/output processing."""
    def __init__(self):
        """Initialize with None - will set sizes after loading robot data."""
        self.snns = []
        self.grid_width = None
        self.grid_height = None
        self.indices = None
        self.types = None
        self.neighbors = None
        self.NUM_SNN = 0  # Number of spiking neural networks (actuators)
        self.update_constants()
    
    def update_constants(self):
        self.INP_SIZE = self.NUM_SNN - 1  # Distances to every other actuator
        self.HIDDEN_SIZE = 3
        self.OUTPUT_SIZE = 1
        self.PARAMS_PER_HIDDEN_LAYER = (self.INP_SIZE + 1) * self.HIDDEN_SIZE
        self.PARAMS_PER_OUTPUT_LAYER = (self.HIDDEN_SIZE + 1) * self.OUTPUT_SIZE
        self.PARAMS_PER_SNN = self.PARAMS_PER_HIDDEN_LAYER + self.PARAMS_PER_OUTPUT_LAYER

    def load_robot_config(self, robot_path=ROBOT_DATA_PATH):
        """
        Load robot configuration from JSON file and initialize SNN.
        
        Args:
            robot_path (str): Path to robot JSON configuration file
            
        Returns:
            tuple: (num_actuators, input_size) - Network dimensions
        """
        if not os.path.exists(robot_path):
            raise FileNotFoundError(f"Robot configuration file not found: {robot_path}")
        with open(robot_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Extract grid dimensions
        self.grid_width = data["grid_width"]
        self.grid_height = data["grid_height"]
        # Extract robot data
        robot_key = list(data["objects"].keys())[0]
        robot_data = data["objects"][robot_key]
        # Store robot structure
        self.indices = robot_data["indices"]
        self.types = robot_data["types"]
        self.neighbors = robot_data["neighbors"]
        print(f"types: {self.types}")
        # Count actuators (types 3 and 4)
        self.NUM_SNN = sum(1 for t in self.types if t in [3, 4])
        self.update_constants()
        # Initialize SNN with proper dimensions
        self.snns = [SpikyNet(
            input_size=self.INP_SIZE,
            hidden_size=self.HIDDEN_SIZE,
            output_size=self.OUTPUT_SIZE
        ) for _ in range(self.NUM_SNN)]
        return self.NUM_SNN, self.INP_SIZE
    
    # def find_corners(self):
    #     """
    #     Find corner voxels of the robot structure.
        
    #     Returns:
    #         list: Indices of corner voxels
    #     """
    #     x_coords = [idx % self.grid_width for idx in self.indices]
    #     y_coords = [idx // self.grid_width for idx in self.indices]
    #     min_x, max_x = min(x_coords), max(x_coords)
    #     min_y, max_y = min(y_coords), max(y_coords)
    #     corners = []
    #     for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    #         if ((x == min_x and y == min_y) or
    #             (x == min_x and y == max_y) or
    #             (x == max_x and y == min_y) or
    #             (x == max_x and y == max_y)):
    #             corners.append(self.indices[i])
    #     return sorted(corners)
    
    # def calculate_corner_distances(self):
    #     """
    #     Calculate normalized pairwise distances between robot corners.
        
    #     Returns:
    #         list: Normalized distances to use as SNN input
    #     """
    #     corners = self.find_corners()
    #     distances = []
    #     for i, corner1 in enumerate(corners):
    #         x1 = corner1 % self.grid_width
    #         y1 = corner1 // self.grid_width
    #         for corner2 in corners[i + 1:]:
    #             x2 = corner2 % self.grid_width
    #             y2 = corner2 // self.grid_width
    #             dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    #             distances.append(dist)
    #     max_dist = max(distances)
    #     return [d/max_dist for d in distances]
    
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
        print(outputs)
        return outputs
    
    def save_output_state(self, output_state, output_path):
        """
        Save output state to JSON file.
        
        Args:
            output_state (dict): State to save
            output_path (str): Path to save JSON file
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_state, f, indent=2)


def main():
    """Main function to demonstrate SNN output generation."""
    runner = SNNRunner()
    try:
        # Load robot configuration and initialize SNN
        num_actuators, input_size = runner.load_robot_config()
        print(f"Initialized SNN with {input_size} inputs")
        # Generate random weights for testing
        print(f"params per snn: {runner.PARAMS_PER_SNN}")
        num_weights = runner.PARAMS_PER_SNN * num_actuators
        test_weights = np.random.rand(num_weights)
        runner.set_snn_weights(test_weights)
        # Generate outputs
        inputs = [np.random.uniform(0, 1, input_size) for _ in range(num_actuators)]
        print("\nRunning get_output_state for 100 steps...")
        output_states = runner.get_output_state(inputs)
        for id, output_state in output_states.items():
            print("\nDuty cycle (raw firing frequencies):")
            for node_idx, dc in enumerate(output_state["duty_cycle"]):
                print(f"Output Node {id+1}: {dc:.3f}")
            print(f"\nTarget length (scaled to {MIN_LENGTH}-{MAX_LENGTH}):")
            for node_idx, action in enumerate(output_state["target_length"]):
                print(f"Actuator {id+1}: {action:.3f}")
            # Save output to JSON
        runner.save_output_state(output_states, "snn_outputs.json")
        print("\nSaved outputs to snn_outputs.json")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    main()
