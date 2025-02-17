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
SIMULATION_DELAY = 0.01  # Delay between simulation steps
ROBOT_DATA_PATH = os.path.join("..", "morpho_demo", "world_data", "bestbot.json")


class SNNRunner:
    """Class to handle SNN input/output processing."""
    def __init__(self):
        """Initialize with None - will set sizes after loading robot data."""
        self.snn = None
        self.grid_width = None
        self.grid_height = None
        self.indices = None
        self.types = None
        self.neighbors = None
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
        # Count actuators (types 3 and 4)
        actuator_count = sum(1 for t in self.types if t in [3, 4])
        # Calculate input size based on corner distances
        corner_indices = self.find_corners()
        input_size = len(corner_indices) * (len(corner_indices) - 1) // 2
        # Initialize SNN with proper dimensions
        hidden_size = 2  # Adjustable based on needs
        self.snn = SpikyNet(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=actuator_count
        )
        return actuator_count, input_size
    def find_corners(self):
        """
        Find corner voxels of the robot structure.
        
        Returns:
            list: Indices of corner voxels
        """
        x_coords = [idx % self.grid_width for idx in self.indices]
        y_coords = [idx // self.grid_width for idx in self.indices]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        corners = []
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            if ((x == min_x and y == min_y) or
                (x == min_x and y == max_y) or
                (x == max_x and y == min_y) or
                (x == max_x and y == max_y)):
                corners.append(self.indices[i])
        return corners
    def calculate_corner_distances(self):
        """
        Calculate normalized pairwise distances between robot corners.
        
        Returns:
            list: Normalized distances to use as SNN input
        """
        corners = self.find_corners()
        distances = []
        for i, corner1 in enumerate(corners):
            x1 = corner1 % self.grid_width
            y1 = corner1 // self.grid_width
            for corner2 in corners[i + 1:]:
                x2 = corner2 % self.grid_width
                y2 = corner2 // self.grid_width
                dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances.append(dist)
        max_dist = max(distances)
        return [d/max_dist for d in distances]
    def get_output_state(self, num_steps=100):
        """
        Run SNN with corner distances as input over multiple timesteps.
        
        Args:
            num_steps (int): Number of simulation steps to run
            
        Returns:
            dict: Contains 'continuous_actions' and 'duty_cycles'
        """
        all_outputs = []
        inputs = self.calculate_corner_distances()
        for _ in range(num_steps):
            outputs = self.snn.compute(inputs)
            all_outputs.append(outputs)
            time.sleep(SIMULATION_DELAY)
        duty_cycles = [neuron.duty_cycle() for neuron in self.snn.output_layer.nodes]
        scale_factor = MAX_LENGTH - MIN_LENGTH
        scaled_actions = [(dc * scale_factor) + MIN_LENGTH for dc in duty_cycles]
        return {
            "continuous_actions": scaled_actions,
            "duty_cycles": duty_cycles
        }
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
        print(f"Initialized SNN with {input_size} inputs and {num_actuators} outputs")
        # Generate random weights for testing
        num_weights = (input_size + 1) * 2 + (2 + 1) * num_actuators
        test_weights = np.random.rand(num_weights).tolist()
        runner.snn.set_weights(test_weights)
        # Generate outputs
        print("\nRunning get_output_state for 100 steps...")
        output_state = runner.get_output_state()
        print("\nDuty Cycles (raw firing frequencies):")
        for node_idx, dc in enumerate(output_state["duty_cycles"]):
            print(f"Output Node {node_idx+1}: {dc:.3f}")
        print(f"\nContinuous Actions (scaled to {MIN_LENGTH}-{MAX_LENGTH}):")
        for node_idx, action in enumerate(output_state["continuous_actions"]):
            print(f"Actuator {node_idx+1}: {action:.3f}")
        # Save output to JSON
        runner.save_output_state(output_state, "snn_outputs.json")
        print("\nSaved outputs to snn_outputs.json")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Error: {str(e)}")
if __name__ == '__main__':
    main()
