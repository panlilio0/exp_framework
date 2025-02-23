"""
Given a genome, runs a simulation of a walking robot in evogym, using an SNN controlled robot,
providing a fitness score corresponding to how far the robot walked.

Author: Thomas Breimer
January 29th, 2025
"""

import os
import cv2
import numpy as np
from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject
from robot.morphology import Morphology
from snn.SNNRunner import SNNRunner

# Simulation constants
ROBOT_SPAWN_X = 3
ROBOT_SPAWN_Y = 10
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
NUM_ITERS = 200
FPS = 50
MODE = "h" # "headless", "screen", or "video"

NUM_ACTUATORS = 4
FITNESS_OFFSET = 100

# Files
ENV_FILENAME = "simple_environment.json"
ROBOT_FILENAME = "speed_bot.json"

def create_video(source, fps=FPS, output_name='output'):
    """
    Saves a video from a list of frames

    Parameters:
        source (list): List of cv2 frames.
        fps (int): Frames per second of video to save.
        output_name (string): Filename of output video.

    """
    current_directory = os.getcwd()
    vid_path = os.path.join(current_directory, "videos", output_name + ".mp4")
    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (source[0].shape[1], source[0].shape[0]))

    for frame in source:
        out.write(frame)
    out.release()

def group_list(flat_list: list, n: int) -> list:
    """
    Groups flat_array into a list of list of size n.

    Parameters:
        flat_list (list): List to groups.
        n: (int): Size of sublists.
    
    Returns:
        list: Grouped list.
    """
    return [list(flat_list[i:i+n]) for i in range(0, len(flat_list), n)]

def run(iters, genome, mode, vid_name=None):
    """
    Runs a single simulation of a given genome.

    Parameters:
        iters (int): How many iterations to run.
        genome (ndarray): The genome of the robot.
        mode (string): How to run the simulation. 
                       "h" runs without any video or visual output.
                       "v" outputs the simulation as a video in the "./videos folder.
                       "s" shows the simulation on screen as a window.
                       "b: shows the simulation on a window and saves a video.
        vid_name (string): If mode is "v" or "b", this is the name of the saved video.
    Returns:
        float: The fitness of the genome.
    """

    if mode in ["v", "b"]: #video or both
        os.makedirs("videos", exist_ok=True)

    # Create world
    world = EvoWorld.from_json(os.path.join('robot', 'world_data', ENV_FILENAME))

    # Add robot
    robot = WorldObject.from_json(os.path.join('robot', 'world_data', ROBOT_FILENAME))

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

    video_frames = []

    # Get position of all robot point masses
    init_raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")

    morphology = Morphology(ROBOT_FILENAME)

    snn_controller = SNNRunner(NUM_ACTUATORS)
    snn_controller.set_snn_weights(genome)

    for i in range(iters):
        # Get point mass locations
        raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")

        # Transform to snn inputs
        snn_inputs = morphology.get_corner_distances(raw_pm_pos)

        # Feed snn and get outputs
        action = snn_controller.compute_all(snn_inputs)

        # Clip actuator target lengths to be between 0.6 and 1.6 to prevent buggy behavior
        action = np.clip(action, ACTUATOR_MIN_LEN, ACTUATOR_MAX_LEN)

        # Set robot action to the action vector. Each actuator corresponds to a vector
        # index and will try to expand/contract to that value
        sim.set_action('robot', action)

        # Execute step
        sim.step()

        if mode == "v":
            video_frames.append(viewer.render(verbose=True, mode="rgb_array"))
        elif mode == "s":
            viewer.render(verbose=True, mode="screen")
        elif mode == "b":
            viewer.render(verbose=True, mode="screen")
            video_frames.append(viewer.render(verbose=True, mode="rgb_array"))

    viewer.close()

    # Get robot point mass posiion position afer sim has run
    final_raw_pm_pos = sim.object_pos_at_time(sim.get_time(), "robot")

    fitness = np.mean(init_raw_pm_pos, 1)[0] - np.mean(final_raw_pm_pos, 1)[0]

    if mode in ["v", "b"]:
        create_video(video_frames, FPS, vid_name)

    return FITNESS_OFFSET - fitness # Turn into a minimization problem

if __name__ == "__main__":
    run(1000, np.array([1]*4*13), "v")
