"""
Simple RMHC of walking robot from scratch in evogym

Author: Thomas Breimer
January 22nd, 2025
"""

import os
import random
import numpy as np
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject
from cmaes import CMA

robot_spawn_x = 3
robot_spawn_y = 10
actuator_min_len = 0.6
actuator_max_len = 1.6
frame_cycle_len = 10
num_actuators = 10
num_iters = 200
mutate_rate = 0.2
fitness_offset = 100

env_file_name = "simple_environment.json"
robot_file_name = "speed_bot.json"

def run_simulation(iters, genome, show=True):
    """
    Runs a single simulation of a given genome.

    Parameters:
        iters (int): How many iterations to run.
        genome (ndarray): The genome of the robot, which is an 
        array of scalars from the robot's average position to the 
        desired length of the muscles.

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

        # Compute the action vector by averaging the avg x & y
        # coordinates and multiplying this scalar by the genome
        action = genome * ((com_1[0] + com_1[1]) / 2)

        # Clip actuator target lengths to be between 0.6 and 1.6 to prevent buggy behavior
        action = np.clip(action, actuator_min_len, actuator_max_len)

        if i % (frame_cycle_len * 2) < frame_cycle_len:
            action = action[0:num_actuators]
        else:
            action = action[num_actuators:(num_actuators * 2)]

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

    return fitness_offset - fitness

if __name__ == "__main__":
    optimizer = CMA(mean=np.ones(20), sigma=2)

    all_solutions = []

    for generation in range(5):
        solutions = []

        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = run_simulation(num_iters, x, False)
            solutions.append((x, value))

        optimizer.tell(solutions)
        print([i[1] for i in solutions])
        print("Generation", generation, "Best Fitness:", solutions[0][1])

        all_solutions.append(solutions)

    best_fitness = fitness_offset
    best_genome = []

    for generation in all_solutions:
        if generation[0][1] < best_fitness:
            best_fitness = generation[0][1]
            best_genome = generation[0][0]

    print("Final Best Fitness", best_fitness)

    run_simulation(num_iters, best_genome)
