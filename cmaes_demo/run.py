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

def run_rmhc(gens, show=True):
    """
    Run a RMHC in evogym. 

    Fitness is defined as the sum of the distance that the 
    robot's center of mass moves in between each step of the simulation.
    The robot morphology is predefined and imported from "speed_bot.json". 
    It has ten actuators. The robot's genome is a vector of twenty scalars. 
    At each time step, the x & y coordinates of the robot's center of mass 
    are averaged and this scalar is multiplied by the genome vector. In a loop, 
    for ten frames, this value is multiplied by the first ten values of the vector. 
    For the next ten frames, they are multipled by the second ten. These values are 
    then clipped to between 0.6 and 1.6 and each represent an actuator target length 
    for that simulation step. The genome is mutated by replacing a vector index with 
    a random float between 0 and 1 with a 20% probabillity for each index.
    
    Parameters:
        iters (int): How many generations to run.
        show (bool): If true, renders all simulations. If false, only renders the fittest robot.
    
    Returns:
        (ndarray, fitness): The fittest genome the RMHC finds and its fitness.
    """

    iters = 100
    genome = np.random.rand(20)
    best_fitness = run_simulation(iters, genome, show)

    print("Starting fitness:", best_fitness)

    for i in range(gens):

        # Mutate
        mutated_genome = genome.copy()

        mutated_genome = np.array([random.random() if random.random() < 0.2 else x for x in mutated_genome])

        new_fitness = run_simulation(iters, mutated_genome, show)

        # Replace old genome with new genome if it is fitter
        if new_fitness > best_fitness:
            print("Found better after", i, "generations:", new_fitness)
            best_fitness = new_fitness
            genome = mutated_genome

    # Show fittest genome
    print("Final fitness", best_fitness)
    run_simulation(500, genome)

    return (genome, best_fitness)


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
    world = EvoWorld.from_json(os.path.join('world_data', 'simple_environment.json'))

    # Add robot
    robot = WorldObject.from_json(os.path.join('world_data', 'speed_bot.json'))

    world.add_from_array(
        name='robot',
        structure=robot.get_structure(),
        x=3,
        y=10,
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
        action = np.clip(action, 0.6, 1.6)

        if i % 20 < 10:
            action = action[0:10]
        else:
            action = action[10:20]

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


    return fitness

if __name__ == "__main__":
    run_rmhc(50, False)
