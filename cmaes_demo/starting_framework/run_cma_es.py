"""
Run cma-es of walking robot in evogym

Author: Thomas Breimer
January 29th, 2025
"""

import os
import pathlib
import datetime
import numpy as np
import pandas as pd
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject
from cmaes import CMA

robot_spawn_x = 3
robot_spawn_y = 10
actuator_min_len = 0.6
actuator_max_len = 1.6
num_actuators = 10
num_iters = 200
num_gens = 50
fitness_offset = 100
fps = 50

avg_freq = 0.1
avg_amp = 1
avg_phase_off = 0
sigma = 2

env_file_name = "simple_environment.json"
robot_file_name = "speed_bot.json"

def sine_wave(time, frequency, amplitude, phase_offset):
  """
  Calculates the sine value at a given time with specified parameters.

  Args:
    time: Time at which to calculate the sine value, in seconds.
    frequency: Frequency of the sine wave in Hz.
    amplitude: Amplitude of the sine wave.
    phase_offset: Phase offset in radians.

  Returns:
    The sine value at the given time.
  """

  angular_frequency = 2 * np.pi * frequency
  return amplitude * np.sin(angular_frequency * time + phase_offset)

def run_simulation(iters, genome, show=True):
    """
    Runs a single simulation of a given genome.

    Parameters:
        iters (int): How many iterations to run.
        genome (ndarray): The genome of the robot.

    Returns:
        float: The fitness of the genome.
    """

    # Create world
    world = EvoWorld.from_json(os.path.join('world_data', env_file_name))

    # Add robot
    robot = WorldObject.from_json(os.path.join('world_data', robot_file_name))

    world.add_from_array(
        name='robot',
        structure=robot.get_structure(),
        x=robot_spawn_x,
        y=robot_spawn_y,
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

        time = i / fps

        # Compute the action vector by taking the first three
        # elements of the genome and inputing them as the frequency,
        # amplitude and phase offset for a sin function at the
        # given time, and using this value for the first actuator,
        # and so on for all actuators
        action = [sine_wave(time, genome[j], genome[j+1], genome[j+2]) for j in range(0, len(genome), 3)]

        # Clip actuator target lengths to be between 0.6 and 1.6 to prevent buggy behavior
        action = np.clip(action, actuator_min_len, actuator_max_len)

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

    return fitness_offset - fitness # Turn into a minimization problem

def run_cma_es(gens):
    """
    Runs the cma_es algorithm on the robot locomotion problem,
    with sin-like robot actuators.

    Parameters:
        gens (int): How many generations to run.
    """
    
    # Generate Results DF
    df_cols = ['Generation', 'Individual', 'Fitness']

    for i in range(num_actuators):
        df_cols = df_cols + ['frequency' + str(i), 'amplitude' + str(i), 'phase_offset' + str(i)]

    df = pd.DataFrame(columns=df_cols)

    optimizer = CMA(mean=np.array([avg_freq, avg_amp, avg_phase_off] * num_actuators), sigma=sigma)

    for generation in range(gens):
        solutions = []

        for indv_num in range(optimizer.population_size):
            x = optimizer.ask()
            value = run_simulation(num_iters, x, False)
            solutions.append((x, value))
            to_add = [generation, indv_num, value] + list(x)
            df.loc[len(df)] = to_add


        optimizer.tell(solutions)
        print([i[1] for i in solutions])
        print("Generation", generation, "Best Fitness:", solutions[0][1])

    # Save csv
    this_dir = pathlib.Path(__file__).parent.resolve()    
    df.to_csv(os.path.join(this_dir, 'out', str(datetime.datetime.now()) + '_run.csv'), index=False)


if __name__ == "__main__":
    run_cma_es(num_gens)
