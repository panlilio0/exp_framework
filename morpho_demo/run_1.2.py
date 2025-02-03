"""
Simple RMHC of walking robot from scratch in evogym

Author: Thomas Breimer, Matthew Meek
January 22nd, 2025
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject

robot_spawn_x = 3
robot_spawn_y = 10
actuator_min_len = 0.6
actuator_max_len = 1.6
frame_cycle_len = 10
num_actuators = 5 #8 for walkbot4billion
num_iters = 100
mutate_rate = 0.2

env_file_name = "simple_environment_long.json"
robot_file_name = "walkbot4billion_reduced.json"
generations = 1500 

exper_directory = 'score_plots/'+ robot_file_name[:-5] + " " + time.asctime() #directory generated for a single run of the program. Stores outputs.


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

    os.mkdir(exper_directory) #generate special directory for results.

    iters = num_iters
    genome = np.random.rand(num_actuators * 2)
    best_fitness = run_simulation(iters, genome, show)

    print("Starting fitness:", best_fitness)
    fitness_by_gen = np.array([]) #array of fitness
    best_fitness_by_gen = np.array([]) #array of fitness (best as of i gen)

    for i in range(gens):

        # Mutate
        mutated_genome = genome.copy()

        mutated_genome = np.array([random.random() if random.random() < mutate_rate else x for x in mutated_genome])

        new_fitness = run_simulation(iters, mutated_genome, show)

        fitness_by_gen = np.append(fitness_by_gen, new_fitness)

        # Replace old genome with new genome if it is fitter
        if new_fitness > best_fitness:
            print("Found better after", i, "generations:", new_fitness)
            best_fitness = new_fitness
            genome = mutated_genome

        best_fitness_by_gen = np.append(best_fitness_by_gen, best_fitness)


    # Show fittest genome
    print("Final fitness", best_fitness)
    run_simulation(num_iters * 5, genome, fittest=True)
    plot_scores(fitness_by_gen, best_fitness_by_gen, gens)

    return (genome, best_fitness)

def run_simulation(iters, genome, show=True, fittest=False): #if fittest, then track action
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

    action_list = []

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

        #action tracking for fittest
        if fittest:
            print(action)
            action_list.append(action)

        # Get robot position after the step
        pos_2 = sim.object_pos_at_time(sim.get_time(), "robot")

        # Compute reward, how far the robot moved in that time step
        com_2 = np.mean(pos_2, 1)
        reward = com_2[0] - com_1[0]
        fitness += reward

        if show:
            viewer.render('screen', verbose=True)
    viewer.close()

    if fittest:
        action_array = np.asarray(action_list)
        plot_action(action_array)
    
    return fitness


def plot_scores(fit_func_scores, fit_func_scores_best, gen_number):

    #make plot
    gen_array = np.arange(1, gen_number+1, 1)
    plottitle = robot_file_name + " scores"
    plt.title(plottitle)
    plt.ylabel("scores")
    plt.xlabel("generations")
    plt.plot(gen_array, fit_func_scores, label= "at gen")
    plt.plot(gen_array, fit_func_scores_best, label= "best by gen")
    plt.legend(loc='upper center')
    plt.savefig(exper_directory +"/"+ plottitle + '.png')
    plt.close()


def plot_action(action_arrays):

    stepcount_array = np.arange(1, len(action_arrays)+1, 1)
    plottitle = robot_file_name + " actions"
    plt.title(plottitle)
    plt.ylabel("target length")
    plt.xlabel("steps")
    
    voxels_list = []
    for i in range(len(action_arrays[0])): #for each voxel
        voxel_action = np.array([])
        for j in range(len(action_arrays)): #for each step
            voxel_action = np.append(voxel_action, action_arrays[j][i]) #I use both i and j, so this is okay, right?
        #plot_voxel(stepcount_array, voxel_action, i)
        voxels_list.append(voxel_action)
        plt.plot(stepcount_array, voxel_action, label= "voxel: " + str(i))

    plt.legend(loc='upper center')
    plt.savefig(exper_directory + "/" + plottitle + '.png')
    plt.close()

    #Save voxel data for later examination
    df = pd.DataFrame(voxels_list)
    df.to_excel(exper_directory +"/"+ plottitle +'.xlsx', index=False)

    #plot individual voxels
    for i in range(len(voxels_list)): #I USE i!!!!!
        plot_voxel(stepcount_array, voxels_list[i], i)


def plot_voxel(stepcount_array, voxel_action, voxel_num):
    plottitle = robot_file_name + " voxel: " + str(voxel_num)
    plt.title(plottitle)
    plt.ylabel("target length")
    plt.xlabel("steps")
    plt.plot(stepcount_array, voxel_action, label= "voxel: " + str(voxel_num))
    plt.legend(loc='upper center')
    plt.savefig(exper_directory +"/"+ plottitle + '.png')
    plt.close()

if __name__ == "__main__":
    run_rmhc(generations, False)


# TWO PROOFS:
# "can" action and replay it sped up vs normal. Robot should act totally different
# Robot with "dead" actuators should behave different from normal healthy robot.