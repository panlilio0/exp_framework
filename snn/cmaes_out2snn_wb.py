import numpy as np

"""
Some functions are assumed to be imported from whichever file it is in - will be modified later once rest of the project is caught up
List of such functions:
- pipeline.get_cmaes_out()
- pipeline.get_num_actuators()

Also replace hidden layer size and output size to actual numbers instead of `X` and `Y`
"""


# CONSTANT

# Placeholder numbers - for now used numbers from the example SNN described while talking to John
NUM_SNN =  4  # pipeline.get_num_actuators()     # Number of spiking neural networks (actuators)
INP_SIZE = NUM_SNN - 1                     # Input size - total number of actuators (number of SNNs - 1)
HIDDEN_SIZE = 1  # X
OUTPUT_SIZE = 1  # Y

NUM_WEIGHTS_PER_HIDDEN_NODE = INP_SIZE      # Number of weight parameters per node
NUM_WEIGHTS_PER_OUTPUT_NODE = HIDDEN_SIZE

# Total number of parameters per SNN is the sum of weights and biases.
PARAMS_PER_HIDDEN_LAYER = (NUM_WEIGHTS_PER_HIDDEN_NODE + 1) * HIDDEN_SIZE  # 1 = number of bias in the node
PARAMS_PER_OUTPUT_LAYER = (NUM_WEIGHTS_PER_OUTPUT_NODE + 1) * OUTPUT_SIZE  # 1 = number of bias in the node
PARAMS_PER_SNN = PARAMS_PER_HIDDEN_LAYER + PARAMS_PER_OUTPUT_LAYER
CMAES_OUT_SIZE = NUM_SNN * PARAMS_PER_SNN

# Mapping voxel (SNN) to SNN parameters in the `snn_param` list
# VOXEL_SNN_MAPPING = {
#     1: "voxel a",
#     2: "voxel b",
#     3: "voxel c",
#     ......
#     N: "voxel N"
# }


def unpack_cmaes_output(cmaes_out):
    """
    Retrieve the flat CMA-ES output and reshape it into a structured format.

    Returns:
        list of dict: A list where each element corresponds to one SNN's parameters.
                      Each dictionary has two keys:
                        - 'weights': A numpy array containing the weight parameters.
                        - 'biases' : A numpy array containing the bias parameters.
                        
    Raises:
        ValueError: If the length of the CMA-ES output does not match the expected size.
    """

    flat_vector = np.array(cmaes_out)  # np.array(pipeline.get_cmaes_out())

    if flat_vector.size != CMAES_OUT_SIZE:
        raise ValueError(f"Expected CMA-ES output vector of size {CMAES_OUT_SIZE}, got {flat_vector.size}.")

    # Reshape the flat vector to a 2D array: each row corresponds to one SNN.
    reshaped = flat_vector.reshape((NUM_SNN, PARAMS_PER_SNN))
    
    # For each SNN, split the parameters into weights and biases.
    snn_parameters = {}
    for snn_idx, params_per_snn in enumerate(reshaped):
        hidden_params = []
        current_node = 0
        for _ in range(HIDDEN_SIZE):
            offset = current_node + NUM_WEIGHTS_PER_HIDDEN_NODE
            hidden_weights = params_per_snn[current_node:offset]
            hidden_biases = params_per_snn[offset]
            hidden_params.append({
                'weights': hidden_weights,
                'bias': hidden_biases
            })
            current_node += PARAMS_PER_HIDDEN_LAYER
        
        output_params = []
        for _ in range(OUTPUT_SIZE):
            offset = current_node + NUM_WEIGHTS_PER_OUTPUT_NODE
            output_weights = params_per_snn[current_node:offset]
            output_biases = params_per_snn[offset]
            output_params.append({
                'weights': output_weights,
                'bias': output_biases
            })
            current_node += PARAMS_PER_OUTPUT_LAYER
        
        snn_parameters[snn_idx] = {
            'hidden_layer': hidden_params,
            'output_layer': output_params
        }
    
    return snn_parameters


if __name__ == '__main__':
    try:
        snn_params = unpack_cmaes_output(np.random.randint(1, 15, CMAES_OUT_SIZE))
        for snn_id, params in snn_params.items():
            print(f"SNN {snn_id}:")

            print("    Hidden Layer:")
            for i, node in enumerate(params['hidden_layer']):
                print(f"        Node {i}: Weights = {node['weights']}; Bias = {node['bias']}")
            
            print("    Output Layer:")
            for i, node in enumerate(params['output_layer']):
                print(f"        Node {i}: Weights = {node['weights']}; Bias = {node['bias']}")
            
            print('\n')
    except Exception as error:
        print(f"Error while unpacking: {error}")