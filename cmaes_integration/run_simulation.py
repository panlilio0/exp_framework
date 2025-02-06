"""
Run cma-es of a sin-based walking robot in evogym.
Accepts two command line arguments: number of generations to run, and sigma.
Example: `python3 run_cma_es.py 50 2`

Author: Thomas Breimer
January 29th, 2025
"""

import os
import numpy as np
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject

# Simulation constants
ROBOT_SPAWN_X = 3
ROBOT_SPAWN_Y = 10
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
NUM_ITERS = 200
FPS = 50

# Starting sin wave characteristics
AVG_FREQ = 0.1
AVG_AMP = 1
AVG_PHASE_OFFSET = 0

NUM_ACTUATORS = 10

# CMA-ES constants
FITNESS_OFFSET = 100
NUM_GENS = 10
SIGMA = 2

# Files
ENV_FILENAME = "simple_environment.json"
ROBOT_FILENAME = "speed_bot.json"

def sine_wave(sin_time, frequency, amplitude, phase_offset):
    """
    Calculates the sine value at a given time with specified parameters.

    Args:
        sin_time (float): Time at which to calculate the sine value, in seconds.
        frequency (int): Frequency of the sine wave in Hz.
        amplitude (float): Amplitude of the sine wave.
        phase_offset (int): Phase offset in radians.

    Returns:
        float: The sine value at the given time.
    """

    angular_frequency = 2 * np.pi * frequency
    return amplitude * np.sin(angular_frequency * sin_time + phase_offset)

def run(iters, genome, show=True):
    """
    Runs a single simulation of a given genome.

    Parameters:
        iters (int): How many iterations to run.
        genome (ndarray): The genome of the robot.

    Returns:
        float: The fitness of the genome.
    """

    # Create world
    world = EvoWorld.from_json(os.path.join('world_data', ENV_FILENAME))

    # Add robot
    robot = WorldObject.from_json(os.path.join('world_data', ROBOT_FILENAME))

    world.add_from_array(
        name='robot',
        structure=robot.get_structure(),
        x=ROBOT_SPAWN_X,
        y=ROBOT_SPAWN_Y,
        connections=robot.get_connections())

    # Create simulation
    sim = EvoSim(world)
    sim.reset()

    # Set up viewer
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')

    fitness = 0

    for i in range(iters):

        # Get position of all robot voxels
        pos_1 = sim.object_pos_at_time(sim.get_time(), "robot")

        # Get mean of robot voxels
        com_1 = np.mean(pos_1, 1)

        sin_time = i / FPS

        # Compute the action vector by taking the first three
        # elements of the genome and inputing them as the frequency,
        # amplitude and phase offset for a sin function at the
        # given time, and using this value for the first actuator,
        # and so on for all actuators
        action = [sine_wave(sin_time, genome[j], genome[j+1], genome[j+2])
                  for j in range(0, len(genome), 3)]

        # Clip actuator target lengths to be between 0.6 and 1.6 to prevent buggy behavior
        action = np.clip(action, ACTUATOR_MIN_LEN, ACTUATOR_MAX_LEN)

        # Set robot action to the action vector. Each actuator corresponds to a vector
        # index and will try to expand/contract to that value
        sim.set_action('robot', action)

        # Execute step
        sim.step()

        # Get robot position after the step
        pos_2 = sim.object_pos_at_time(sim.get_time(), "robot")

        # Compute reward, how far the robot moved in that time step
        com_2 = np.mean(pos_2, 1)
        reward = com_2[0] - com_1[0]
        fitness += reward

        if show:
            viewer.render('screen', verbose=True)

    viewer.close()

    return FITNESS_OFFSET - fitness # Turn into a minimization problem