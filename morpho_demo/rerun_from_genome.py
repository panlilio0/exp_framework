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
import argparse
import json
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject

ROBOT_SPAWN_X = 3
ROBOT_SPAWN_Y = 10
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
FRAME_CYCLE_LEN = 10
#NUM_ACTUATORS = 8  #8 for walkbot4billion
#NUM_ITERS = 100
MUTATE_RATE = 0.2

ENV_FILENAME = "simple_environment_long.json"
#ROBOT_FILENAME = "walkbot4billion.json"
#GENERATIONS = 1500

#EXPER_DIR = 'score_plots/' + ROBOT_FILENAME[:-5] + " " + time.asctime(
#)  #directory generated for a single run of the program. Stores outputs.


def retrieve_actuator_count(robot_filename):
    jsonpath = os.path.join('world_data', robot_filename)
    jsonf = open(jsonpath)
    data = json.load(jsonf)
    actuator_count = 0
    key_for_robot = list(data["objects"].keys())[0] # No, I'm not happy about it, either.
    # The robot's key in the json dicts can vary because the design tool is stupid. So this is a fix.
    for voxel in data["objects"][key_for_robot]["types"]: 
        if voxel == 3 or voxel == 4: # 3 and 4 represent horizontal and vertical actuators.
            actuator_count += 1 

    return actuator_count


def retrieve_genome(genome_filename, gen):
    '''
    Gets the genome from the genome csv file

    Parameters:
        genome_filename (str): Location of robot genome csv.
        gen (int): generation number of the robot.
        
    Returns:
        genome (ndarray): The genome of the robot, which is an 
        array of scalars from the robot's average position to the 
        desired length of the muscles.
    '''

    dfpath = os.path.join('genomes', genome_filename)
    df = pd.read_csv(dfpath)
    df_cropped = df.drop(['generation','best_fitness'], axis=1) #gee thanks cmaes team.
    target_genome = df_cropped.iloc[gen].to_numpy()

    print(target_genome)
    return target_genome

def run_simulation(iters, gen,
                   robot_filename,
                   genome_filename):  
    """
    Runs a single simulation of a given genome.

    Parameters:
        iters (int): How many iterations to run.
        genome_filename (str): Location of robot genome csv.
        robot_filename (str): Location of robot json.

    Returns:
        float: The fitness of the genome.
    """

    # Create world
    world = EvoWorld.from_json(os.path.join('world_data', ENV_FILENAME))

    # Add robot
    robot = WorldObject.from_json(os.path.join('world_data',robot_filename))

    world.add_from_array(name='robot',
                         structure=robot.get_structure(),
                         x=ROBOT_SPAWN_X,
                         y=ROBOT_SPAWN_Y,
                         connections=robot.get_connections())

    #get actuators
    num_actuators = retrieve_actuator_count(robot_filename)

    #get genome
    genome = retrieve_genome(genome_filename, gen)

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
            action = action[0:num_actuators]
        else:
            action = action[num_actuators:(num_actuators * 2)]

        # Set robot action to the action vector. Each actuator corresponds to a vector
        # index and will try to expand/contract to that value
        sim.set_action('robot', action)

        # Execute step
        sim.step()

        #action tracking
        print(action)

        # Get robot position after the step
        pos_2 = sim.object_pos_at_time(sim.get_time(), "robot")

        # Compute reward, how far the robot moved in that time step
        com_2 = np.mean(pos_2, 1)
        reward = com_2[0] - com_1[0]
        fitness += reward

        
        viewer.render('screen', verbose=True)
    viewer.close()

    return fitness


if __name__ == "__main__":

    #args 
    parser = argparse.ArgumentParser(description="Arguments for simple experiment.")
    parser.add_argument("robot_filename", type=str, help=
                        "The json file of the robot you want to run the experiment with.")
    parser.add_argument("genome_filename", type=str, help=
                        "The csv file of genomes of the robot you want to run the experiment with.")
    parser.add_argument("gen", type=int, help=
                        "The generation of the genome you want to run.")
    parser.add_argument("--iters", default=400, type=int, 
                        help="number of iterations in the simulation.")

    args = parser.parse_args()

    fitness = run_simulation(args.iters, args.gen, args.robot_filename, args.genome_filename)
    print("Fitness: " + str(fitness))

#python rerun_from_genome.py speed_bot.json sample_output.csv 5
#python rerun_from_genome.py bestbot.json main_bot_50gens.csv 31