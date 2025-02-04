"""
CMA-ES implementation of searching for a walking robot in evogym.
Accepts two command line arguments: number of generations to run, and sigma.
Example: `python3 run.py 50 2`

Author: Thomas Breimer
February 3rd, 2025
"""

import os
import sys
import numpy as np
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject
from cmaes import CMA

# Simulation constants
ROBOT_SPAWN_X = 3
ROBOT_SPAWN_Y = 10
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
NUM_ITERS = 200

# Robot characteristics
FRAME_CYCLE_LEN = 10
NUM_ACTUATORS = 10

# CMA-ES constants
FITNESS_OFFSET = 100
NUM_GENS = 10
SIGMA = 2

# Files
ENV_FILENAME = "simple_environment.json"
ROBOT_FILENAME = "speed_bot.json"

def run_simulation(iters, genome, show=True):
    """
    Runs a single simulation of a given genome.

    Parameters:
        iters (int): How many iterations to run.
        genome (ndarray): The genome of the robot, which is an 
        array of scalars from the robot's average position to the 
        desired length of the muscles.
        show (bool): Runs in headless mode when False.

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

        # Compute the action vector by averaging the avg x & y
        # coordinates and multiplying this scalar by the genome
        action = genome * ((com_1[0] + com_1[1]) / 2)

        # Clip actuator target lengths to be between 0.6 and 1.6 to prevent buggy behavior
        action = np.clip(action, ACTUATOR_MIN_LEN, ACTUATOR_MAX_LEN)

        if i % (FRAME_CYCLE_LEN * 2) < FRAME_CYCLE_LEN:
            action = action[0:NUM_ACTUATORS]
        else:
            action = action[NUM_ACTUATORS:(NUM_ACTUATORS * 2)]

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

    return FITNESS_OFFSET - fitness

def run_cma(gens, sigma):
    """
    Runs cma-es with run_simulation as the fitness function.
    
    Parameters:
        gens (int): Number of generations to run.
        sigma (float): The standard deviation of the normal distribution 
        used to generate new candidate solutions
    """

    # Set up cma-es
    optimizer = CMA(mean=np.ones(NUM_ACTUATORS * 2), sigma=sigma)
    all_solutions = []

    # Run generations
    for generation in range(gens):
        solutions = []

        # Run individuals
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = run_simulation(NUM_ITERS, x, False)
            solutions.append((x, value))

        # Tell CMA-ES about preformance of individuals
        optimizer.tell(solutions)

        # Update logs
        print([i[1] for i in solutions])
        print("Generation", generation, "Best Fitness:", solutions[0][1])
        all_solutions.append(solutions)

    # Find best individual (not nec. in last generation)

    best_fitness = FITNESS_OFFSET
    best_genome = []

    for generation in all_solutions:
        if generation[0][1] < best_fitness:
            best_fitness = generation[0][1]
            best_genome = generation[0][0]

    # Show best individual

    print("Final Best Fitness", best_fitness)

    run_simulation(NUM_ITERS, best_genome)

if __name__ == "__main__":
    args = sys.argv

    if len(args) > 1:
        NUM_GENS = int(args[1])

    if len(args) > 2:
        SIGMA = float(args[2])

    run_cma(NUM_GENS, SIGMA)
