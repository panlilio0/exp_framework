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
NUM_SNN = pipeline.get_num_actuators()     # Number of spiking neural networks (actuators)
INP_SIZE = NUM_SNN - 1                     # Input size - total number of actuators (number of SNNs - 1)
HIDDEN_SIZE = X
OUTPUT_SIZE = Y

NUM_WEIGHTS_PER_HIDDEN_NODE = INP_SIZE      # Number of weight parameters per node
NUM_BIASES_PER_HIDDEN_NODE = 1              # Number of biases per node
NUM_WEIGHTS_PER_OUTPUT_NODE = HIDDEN_SIZE
NUM_BIASES_PER_OUTPUT_NODE = 1

# Total number of parameters per SNN is the sum of weights and biases.
PARAMS_PER_HIDDEN_LAYER = (NUM_WEIGHTS_PER_HIDDEN_NODE + NUM_BIASES_PER_HIDDEN_NODE) * HIDDEN_SIZE
PARAMS_PER_OUTPUT_LAYER = (NUM_WEIGHTS_PER_OUTPUT_NODE + NUM_BIASES_PER_OUTPUT_NODE) * OUTPUT_SIZE
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


def unpack_cmaes_output():
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
    # Assume get_cmaes_out() is provided by another module.
    # For example, one might import it like:
    # from external_module import get_cmaes_out
    
    # Expected to return a flat array-like object. 
    # Function can be added or imported from evogym codebase once the team finishes building it.
    flat_vector = np.array(pipeline.get_cmaes_out())

    if flat_vector.size != CMAES_OUT_SIZE:
        raise ValueError(f"Expected CMA-ES output vector of size {CMAES_OUT_SIZE}, got {flat_vector.size}.")

    # Reshape the flat vector to a 2D array: each row corresponds to one SNN.
    reshaped = flat_vector.reshape((NUM_SNN, PARAMS_PER_SNN))
    
    # For each SNN, split the parameters into weights and biases.
    snn_parameters = []
    for snn in reshaped:
        hidden_params = []
        offset = 0

        for _ in range(HIDDEN_SIZE):
            hidden_weights = snn[offset:(offset + NUM_WEIGHTS_PER_HIDDEN_NODE)]
            hidden_biases = params[NUM_BIASES_PER_HIDDEN_NODE:]
            hidden_params.append()
        
        snn_parameters.append({
            'weights': hidden_weights,
            'biases': hidden_biases
        })
    
    return snn_parameters
