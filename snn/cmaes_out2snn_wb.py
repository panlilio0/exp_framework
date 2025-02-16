"""
Module for handling CMA-ES output for spiking neural networks.
"""

import numpy as np
from john_code_snn import SpikyNet


# Replace hardcoded values with actual values from the morphology.
# Implement the mapping of the SNNs to the voxels in the morphology.


# CONSTANT

# Placeholder numbers - for now used numbers from the example SNN described while talking to John
NUM_SNN =  4                               # Number of spiking neural networks (actuators)
INP_SIZE = NUM_SNN - 1                     # Distances to every other actuator
HIDDEN_SIZE = 3
OUTPUT_SIZE = 1

# Number of weight parameters per node
NUM_WEIGHTS_PER_HIDDEN_NODE = INP_SIZE
NUM_WEIGHTS_PER_OUTPUT_NODE = HIDDEN_SIZE

# Total number of parameters per SNN is the sum of weights and biases (= 1) for each node.
PARAMS_PER_HIDDEN_LAYER = (NUM_WEIGHTS_PER_HIDDEN_NODE + 1) * HIDDEN_SIZE
PARAMS_PER_OUTPUT_LAYER = (NUM_WEIGHTS_PER_OUTPUT_NODE + 1) * OUTPUT_SIZE
PARAMS_PER_SNN = PARAMS_PER_HIDDEN_LAYER + PARAMS_PER_OUTPUT_LAYER
CMAES_OUT_SIZE = NUM_SNN * PARAMS_PER_SNN
print(f"CMAES_OUT_SIZE: {CMAES_OUT_SIZE}")

# Mapping voxel (SNN) to SNN ID in the `snn_param` list
# VOXEL_SNN_MAPPING = {
#     1: "voxel a",
#     2: "voxel b",
#     3: "voxel c",
#     ......
#     N: "voxel N"
# }


def unpack_cmaes_output(cmaes_out):
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

    if flat_vector.size != CMAES_OUT_SIZE:
        raise ValueError(f"Expected CMA-ES output vector of size {CMAES_OUT_SIZE}, \
                         got {flat_vector.size}.")

    # Reshape the flat vector to a 2D array: each row corresponds to one SNN.
    reshaped = flat_vector.reshape((NUM_SNN, PARAMS_PER_SNN))

    # For each SNN, split the parameters into weights and biases.
    snn_parameters = {}
    for snn_idx, params_per_snn in enumerate(reshaped):
        hidden_params = params_per_snn[:PARAMS_PER_HIDDEN_LAYER]
        output_params = params_per_snn[PARAMS_PER_HIDDEN_LAYER:]
        snn_parameters[snn_idx] = {
            'hidden_layer': hidden_params,
            'output_layer': output_params
        }

    return snn_parameters


if __name__ == '__main__':
    def test():
        """
        Test function for `unpack_cmaes_output()`: 
        Unpacks a random CMA-ES output and sets the weights of a SpikyNet.
        """
        try:
            snns = [SpikyNet(input_size=INP_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)\
                    for _ in range(NUM_SNN)]
            print("creating snn_params")
            snn_params = unpack_cmaes_output(np.random.randint(1, 75, CMAES_OUT_SIZE))
            print("created snn_params")
            print(snn_params)
            print("Testing unpack+set:", end='\n')
            for snn_id, params in snn_params.items():
                print(f"SNN {snn_id} params: {params}")
                snns[snn_id].hidden_layer.set_weights(*params['hidden_layer'])
                snns[snn_id].output_layer.set_weights(*params['output_layer'])
                print(f"SNN {snn_id}: ")
                print(snns[snn_id].compute(np.random.randint(1, 15, INP_SIZE)))
                print('\n')
        except Exception as error:
            print(f"Error while unpacking: {error}")

    test()
