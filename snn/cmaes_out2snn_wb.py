"""
Module for handling CMA-ES output for spiking neural networks.
"""

import numpy as np

# CONSTANT

# Placeholder numbers - for now used numbers from the example SNN described while talking to John
NUM_SNNS = 8  # Number of spiking neural networks (actuators)
NUM_WEIGHTS_PER_SNN = 6  # Number of weight parameters per SNN
NUM_BIASES_PER_SNN = 2  # Number of bias parameters per SNN

# Total number of parameters per SNN is the sum of weights and biases.
PARAMS_PER_SNN = NUM_WEIGHTS_PER_SNN + NUM_BIASES_PER_SNN

# Mapping voxel (SNN) to SNN parameters in the `snn_param` list
# VOXEL_SNN_MAPPING = {
#     1: "voxel a",
#     2: "voxel b",
#     3: "voxel c",
#     ......
#     N: "voxel N"
# }

def get_cmaes_out():
    """
    Mock function to simulate getting CMA-ES output.
    
    Returns:
        np.ndarray: A random array simulating the CMA-ES output.
    """
    return np.random.rand(NUM_SNNS * PARAMS_PER_SNN)

def get_expected_cmaes_size():
    """
    Calculate the expected size of the CMA-ES output vector.
    
    Returns:
        int: The total number of parameters (NUM_SNNS * PARAMS_PER_SNN).
    """
    return NUM_SNNS * PARAMS_PER_SNN


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
    flat_vector = np.array(get_cmaes_out())  # Use the mock function

    expected_size = get_expected_cmaes_size()
    if flat_vector.size != expected_size:
        raise ValueError(
            f"Expected CMA-ES output vector of size {expected_size}, got {flat_vector.size}."
        )

    # Reshape the flat vector to a 2D array: each row corresponds to one SNN.
    reshaped = flat_vector.reshape((NUM_SNNS, PARAMS_PER_SNN))

    # For each SNN, split the parameters into weights and biases.
    snn_parameters = []
    for params in reshaped:
        weights = params[:NUM_WEIGHTS_PER_SNN]
        biases = params[NUM_WEIGHTS_PER_SNN:]
        snn_parameters.append({'weights': weights, 'biases': biases})

    return snn_parameters
