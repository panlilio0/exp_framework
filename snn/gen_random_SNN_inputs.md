"""
Generate random inputs for SNN
"""

import random
from john_code_snn import *
import time

"""
5 input neurons 
2 hidden neurons 
4 output neurons
"""
"""
Generate SNN outputs and convert to EvoGym actuator values
"""

import random
import json
from john_code_snn import *

class SNNAdapter:
    def __init__(self):
        # Initialize network with 5 inputs -> 2 hidden -> 4 outputs
        self.snn = SpikyNet(input_size=5, hidden_size=2, output_size=8)
        
    def run_simulation(self, num_steps=100):
        """Run SNN with random inputs over multiple timesteps"""
        all_outputs = []
        
        for _ in range(num_steps):
            # Generate new random inputs each step
            inputs = [random.random() for _ in range(5)]
            
            # Get binary outputs (0/1 spikes)
            outputs = self.snn.compute(inputs)
            all_outputs.append(outputs)
            
            # Let neurons accumulate firelog for duty cycle
            time.sleep(0.01)  # Simulate real-time behavior
            
        # Calculate duty cycles (firing frequency)
        duty_cycles = [neuron.duty_cycle() for neuron in self.snn.output_layer.nodes]
        
        # Scale to actuator range (0.6-1.6)
        scaled_actions = [(dc * 1.0) + 0.6 for dc in duty_cycles]  # 8 values; each actuator needs 1 control value
        
        # Save to JSON
        with open("snn_outputs.json", "w") as f:
            json.dump({
                "continuous_actions": scaled_actions,
                "duty_cycles": duty_cycles
            }, f, indent=2)
            
        return scaled_actions

if __name__ == '__main__':
    adapter = SNNAdapter()
    actions = adapter.run_simulation(num_steps=200)
    print("Final actuator values:", actions)
        
    # Currently our SNN outputs are just binary spikes: 0/1
    
    # ➜ python3 snn/gen_random_SNN_inputs.py
    # current level: 0.0, bias: 0.5198109672015447
    # new level: 0.02137625679920529
    # 0.0

    # So now, we convert the binary spikes to continous spikes so there could be action
    # Converting this to the actuator range of 0.6-1.6
    
    