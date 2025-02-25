from evogym import EvoWorld, EvoSim, EvoViewer
from evogym import WorldObject
import numpy as np
import os

# Simulation constants
ROBOT_SPAWN_X = 3
ROBOT_SPAWN_Y = 1
ACTUATOR_MIN_LEN = 0.6
ACTUATOR_MAX_LEN = 1.6
NUM_ITERS = 100000
FPS = 50

NUM_ACTUATORS = 1

# Files
ENV_FILENAME = "simple_environment.json"
ROBOT_FILENAME = "single_voxel.json"

def run(iters):
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

    # Create world
    world = EvoWorld.from_json(os.path.join(os.path.dirname(__file__), 'world_data/' + ENV_FILENAME))

    # Add robot
    robot = WorldObject.from_json(os.path.join(os.path.dirname(__file__), 'world_data/' + ROBOT_FILENAME))

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

        # Compute the action vector by taking the first three
        # elements of the genome and inputing them as the frequency,
        # amplitude and phase offset for a sin function at the
        # given time, and using this value for the first actuator,
        # and so on for all actuators

        """
        if i % 2 == 0:
            action = [0.6]
        else:
            action = [1.6]
            """
        action = [1.6]

        # Clip actuator target lengths to be between 0.6 and 1.6 to prevent buggy behavior
        action = np.clip(action, ACTUATOR_MIN_LEN, ACTUATOR_MAX_LEN)

        # Set robot action to the action vector. Each actuator corresponds to a vector
        # index and will try to expand/contract to that value
        sim.set_action('robot', action)
        #print(sim.vel_at_time(sim.get_time())) if i % 10 == 0 else None
        #print(sim._object_names) if i % 10 == 0 else None
        #print(sim.object_orientation_at_time(sim.get_time(), "robot")) if i % 10 == 0 else None
        #print(world.translate_object("robot", 0, 0, 0)) if i % 10 == 0 else None
        # Execute step
        sim.step()

        # Get robot position after the step
        pos_2 = sim.object_pos_at_time(sim.get_time(), "robot")

        # Compute reward, how far the robot moved in that time step
        com_2 = np.mean(pos_2, 1)
        reward = com_2[0] - com_1[0]
        fitness += reward

        viewer.render('screen', verbose=True)
    
    viewer.close()


if __name__=="__main__":
    run(NUM_ITERS)

