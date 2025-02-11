"""
Given a genome, runs a simulation of a walking robot in evogym, providing a fitness score 
corresponding to how far the robot walked.

Author: Thomas Breimer
January 29th, 2025
"""

import os
import cv2
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
MODE = "headless" # "headless", "screen", or "video"

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

def run(iters, genome, mode, vid_name=None):
    """
    Runs a single simulation of a given genome.

    Parameters:
        iters (int): How many iterations to run.
        genome (ndarray): The genome of the robot.
        mode (string): How to run the simulation. 
                       "headless" runs without any video or visual output.
                       "video" outputs the simulation as a video in the "./videos folder.
                       "screen" shows the simulation on screen as a window.
                       "both: shows the simulation on a window and saves a video.
        vid_name (string): If mode is "video" or "both", this is the name of the saved video.
    Returns:
        float: The fitness of the genome.
    """

    if mode in ["video", "both"]:
        os.makedirs("videos", exist_ok=True)

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

    video_frames = []

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

        if mode == "video":
            video_frames.append(viewer.render(verbose=True, mode="rgb_array"))
        elif mode == "screen":
            viewer.render(verbose=True, mode="screen")
        elif mode == "both":
            viewer.render(verbose=True, mode="screen")
            video_frames.append(viewer.render(verbose=True, mode="rgb_array"))

    viewer.close()

    if mode in ["video", "both"]:
        create_video(video_frames, FPS, vid_name)

    return FITNESS_OFFSET - fitness # Turn into a minimization problem
