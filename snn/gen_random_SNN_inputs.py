"""
Generate random inputs for SNN
"""

import random
from john_code_snn import *

"""
5 input neurons 
2 hidden neurons 
4 output neurons
"""
class generate_random_inputs:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def get_random() -> list:
        random_inputs = [random.random() for _ in range(5)]
        return random_inputs


if __name__ == '__main__':
    random_snn_inputs = generate_random_inputs.get_random()
    new_spiky_node = SpikyNode(5)
    snn_outputs = new_spiky_node.compute(random_snn_inputs)

    print(snn_outputs)
    
    # Currently our SNN outputs are just binary spikes: 0/1
    
    # ➜ python3 snn/gen_random_SNN_inputs.py
    # current level: 0.0, bias: 0.5198109672015447
    # new level: 0.02137625679920529
    # 0.0

    # So now, we convert the binary spikes to continous spikes so there could be action
    # Converting this to the actuator range of 0.6-1.6
    
    