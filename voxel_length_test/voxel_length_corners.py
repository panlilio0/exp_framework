"""
Simple RMHC of walking robot from scratch in evogym

Author: Thomas Breimer, Matthew Meek
January 22nd, 2025

Edited By: James Gaskell
February 25th, 2025
"""

import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject
from special_classes import corner as corn
from pathlib import Path
from datetime import datetime
import csv

ROBOT_SPAWN_X = 2
ROBOT_SPAWN_Y = 1
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
FRAME_CYCLE_LEN = 10
MUTATE_RATE = 0.2

ENV_FILENAME = "simple_environment_long.json"
ROBOT_FILENAME = "single_voxel.json"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATE_TIME = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')


def retrieve_actuator_count(robot_filename):
    '''
    Retrieves the number of actuator voxels the given robot has.

    Parameters:
        robot_filename (str): Location of robot json.

    Returns:
        An int: the number of actuators in the robot.
    '''
    jsonpath = os.path.join(os.path.join(os.path.dirname(__file__), 'world_data/' + ROBOT_FILENAME))
    jsonf = open(jsonpath)
    data = json.load(jsonf)
    actuator_count = 0
    key_for_robot = list(data["objects"].keys())[0] # No, I'm not happy about it, either.
    # The robot's key in the json dicts can vary because the design tool is stupid.
    # So this is a fix.
    for voxel in data["objects"][key_for_robot]["types"]:
        if voxel in (3, 4): # 3 and 4 represent horizontal and vertical actuators.
            actuator_count += 1

    return actuator_count


def run_simulation(iters,
                   corners,
                   show=True,
                   fittest=False):  #if fittest, then track action
    """
    Runs a single simulation of a given genome.

    Parameters:
        iters (int): How many iterations to run.
        genome (ndarray): The genome of the robot, which is an 
        array of scalars from the robot's average position to the 
        desired length of the muscles.
        robot_filename (str): Location of robot json.
        show (bool): Whether or not to display a simulation of the robot.
        fittest (bool):  Whether or not the robot is the fittest if its generation

    Returns:
        float: The fitness of the genome.
    """

    # Create world
    world = EvoWorld.from_json(os.path.join(os.path.dirname(__file__), 'world_data/' + ENV_FILENAME))

    # Add robot
    robot = WorldObject.from_json(os.path.join(os.path.dirname(__file__), 'world_data/' + ROBOT_FILENAME))

    world.add_from_array(name='robot',
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

    for i in range(iters):

        # Get position of all robot point masses
        pos_1 = sim.object_pos_at_time(sim.get_time(), "robot")


        # CORNERS TELEMETRY STUFF
        # SNN SQUAD, YOUR DATA IS HERE
        # CHANGE THE PRINT TO BE WHATEVER YOUR STUFF NEEDS!
        distances = get_all_distances(pos_1, corners)

        # Plot the corner distances to record when the robot stops expanding/contracting
        record_distances(distances)

        # if i < 10:
            # print("Corners dists at iter " + str(i) + " :\n" + str(distances) +"\n")
        # and also probably mess with the genome stuff or something idk
        # /CORNERS TELEMETRY STUFF


        # Compute the action vector by averaging the avg x & y
        # coordinates and multiplying this scalar by the genome

        action = np.full(
                        shape=retrieve_actuator_count(ROBOT_FILENAME),
                        fill_value=1.6,
                        dtype=np.float64
                        )

        # Clip actuator target lengths to be between 0.6 and 1.6 to prevent buggy behavior
        action = np.clip(action, ACTUATOR_MIN_LEN, ACTUATOR_MAX_LEN)

        # Set robot action to the action vector. Each actuator corresponds to a vector
        # index and will try to expand/contract to that value
        if i == 0:
            sim.set_action('robot', action)

        # Execute step
        sim.step()

        # Get robot position after the step
        if show:
            viewer.render('screen', verbose=True)
    viewer.close()


def find_corners():
    '''
    Finds the corners, returns them in an array of four corner objects.

    WARNING:
    This function expects a robot whose outermost corners are at the 
    minimum and maximum x and y values. 
    IT WILL BREAK IF THIS IS NOT THE CASE!!!

    Parameters:
        robot_filename (string): name of the json file the robot lives in.

    Returns:
        An ndarray of four corner objects. Each respresents an outer corner.
    '''
    # Stupid little hack but it works like magic.
    # We create a 0-time "simulation" to get coords for point masses.
    # Using those, we can figure out the indicies for the corners.
    world = EvoWorld.from_json(os.path.join(os.path.dirname(__file__), 'world_data/' + ENV_FILENAME))
    robot = WorldObject.from_json(os.path.join(os.path.dirname(__file__), 'world_data/' + ROBOT_FILENAME))
    world.add_from_array(name='robot',
                         structure=robot.get_structure(),
                         x=ROBOT_SPAWN_X,
                         y=ROBOT_SPAWN_Y,
                         connections=robot.get_connections())
    sim = EvoSim(world)
    sim.reset()
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    xy_coords = sim.object_pos_at_time(sim.get_time(), "robot")
    viewer.close()

    # xy_coords is an ndarray of ndarrays
    # first array is x, second array is y.
    x_min = xy_coords[0].min()
    y_min = xy_coords[1].min()
    x_max = xy_coords[0].max()
    y_max = xy_coords[1].max()

    top_right = corn.Corner(find_pm_index(xy_coords, x_max, y_max))
    top_left = corn.Corner(find_pm_index(xy_coords, x_min, y_max))
    bottom_right = corn.Corner(find_pm_index(xy_coords, x_max, y_min))
    bottom_left = corn.Corner(find_pm_index(xy_coords, x_min, y_min))

    toreturn = np.array([top_right, top_left, bottom_right, bottom_left])

    return toreturn


def find_pm_index(xy_coords, x_target, y_target):
    '''
    Finds the index of the point mass at the given x and y coords.

    Parameters:
        xy_coords (ndarray): array containg all of the point mass coords.
        x_target (int): x coord of desired point mass.
        y_target (int): y coord of desired point mass

    Returns:
        int index in xy_coords of the point mass with the desired x and y coords
    '''
    for i in range(len(xy_coords[0])): # i is the point, so this isn't a list comp thing.
        if xy_coords[0][i] == x_target and  xy_coords[1][i] == y_target:
            return i
    raise ValueError("No point mass with target coords.")


def get_all_distances(xy_coords, corners):
    '''
    Gets all of the distances between all corners.
    Excludes 0s (distance from corner to itself) and repeats. 

    Parameters:
        xy_coords (ndarray): array containg all of the point mass coords.
        corners (ndarray): An array of corner objects.

    Returns:
        A 1D ndarray of floats, the distances between the corners. 
        Array should be of len (n-1)th triangular number, where n is the len of corners
        (that is, the number of corners in corners)

    '''

    all_distances = []

    # Nested for loops?!
    # BARF!
    for i in range(len(corners)): #use i to avoid inserting duplicate
        c_distances = corners[i].get_corner_distances(xy_coords, corners)
        #print(str(corners[i]) + " to all: " + str(c_distances))
        for distance in c_distances[i+1:]: #excludes 0s and pre-existing values
            all_distances.append(distance)

    return np.array(all_distances)

def record_distances(distances):
    '''
    Records the distances between the corners in a csv file.
    
    Parameters: 
        distances (ndarray): The distances between the corners.
    '''

    with open(Path(os.path.join(ROOT_DIR, "distance_outputs/" + DATE_TIME + ".csv")), "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(distances)

def make_output_file():
    '''
    Makes the output file for the corner distances.
    Appends the header to the csv file
    '''

    Path(os.path.join(ROOT_DIR, "distance_outputs")).mkdir(parents=True, exist_ok=True)
    csv_header = ['TR-TL', 'TR-BR', 'TR-BL', 'TL-BR', 'TL-BL', 'BR-BL']

    csv_name = DATE_TIME + ".csv"
    csv_path = os.path.join(ROOT_DIR, "distance_outputs", csv_name)

    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)


if __name__ == "__main__":

    #args
    parser = argparse.ArgumentParser(description="Arguments for simple experiment.")
    parser.add_argument("--robot_filename", type=str, default="single_voxel.json",
                        help="The json file of the robot you want to run the experiment with.")
    parser.add_argument("--iters", default=100, type=int,
                        help="number of iterations per generation.")
    parser.add_argument("--gens", default=1500, type=int,
                        help="number of generations.")

    args = parser.parse_args()
    
    corners = find_corners()

    make_output_file() 

    run_simulation(args.iters, corners, fittest=True)

