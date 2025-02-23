"""
Representation of an evogym robot.

Authors: Thomas Breimer, Matthew Meek
February 21st, 2025
"""

import os
import numpy as np
from evogym import EvoWorld, EvoSim, EvoViewer, WorldObject
from robot.actuator import Actuator
from robot.corner import Corner

ROBOT_SPAWN_X = 0
ROBOT_SPAWN_Y = 10
ENV_FILENAME = "simple_environment.json"

class Morphology:
    """
    Our own internal representation of an evogym robot.
    """

    def __init__(self, filename: str):
        """
        Given an evogym robot file, constructs a robot morphology.

        Parameters:
            filename (str): Filename of the robot .json file.
        """
        
        self.robot_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "world_data", filename)
        self.structure = self.get_structure(self.robot_filepath)
        self.actuators = self.create_actuator_voxels(self.structure)
        self.corners = self.find_corners(self.robot_filepath)

    def get_structure(self, robot_filepath: str) -> np.ndarray:
        """
        Return the robot’s structure matrix.

        Parameters:
            robot_filepath (str): The filename of the robot to get.

        Returns:
            np.ndarray: (n, m) array specifing the voxel structure of the object.
        """

        # Get robot structure
        robot = WorldObject.from_json(robot_filepath)
        return robot.get_structure()

    def create_actuator_voxels(self, structure: np.ndarray) -> list:
        """
        Given a robot structure, creates vertices.

        Parameters:
            structure (np.ndarray): array specifing the voxel structure of the object.

        Returns:
            list: A list of actuator objects.
        """

        # Evogym assigns point mass indices by going through the structure array
        # left to right, top to bottom. The first voxel it sees, it assigns its
        # top left point mass to index zero, top right point mass to index one,
        # bottom left point mass to index two, and bottom right point mass to
        # index three. This pattern continues, expect that any point masses that
        # are shared with another voxel and have already been seen are not added to
        # the point mass array. This script goes through this process, constructing
        # the point mass array and identifying shared point masses to create correct
        # actuator objects.

        # To return, will contain Actuator objects
        actuators = []

        # List of tuples (x, y) corresponding to initial point mass positions and index
        # within this list corresponding the their index when calling robot.get_pos()
        point_masses = []

        # Dimensions of the robot
        height = len(structure)
        length = len(structure[0])

        # Will be the coordinates of the top left point mass of ther current voxel. 
        top_y = height
        left_x = 0

        # Follows a similar pattern to point masses, top right actuator is zero,
        # and increments going left to right then top to bottom down the grid
        actuator_action_index = 0

        for row in structure:
            for voxel_type in row:

                right_x = left_x + 1
                bottom_y = top_y - 1

                # Check if top left point mass already in point_masses
                if (left_x, top_y) in point_masses:
                    # If so, find index will be the index of where it already is in the array
                    top_left_index = point_masses.index((left_x, top_y))
                else:
                    # Else, we make a new point mass position
                    top_left_index = len(point_masses)
                    point_masses.append((left_x, top_y))

                # Repeat for top right point mass
                if (right_x, top_y) in point_masses:
                    top_right_index = point_masses.index((right_x, top_y))
                else:
                    top_right_index = len(point_masses)
                    point_masses.append((right_x, top_y))

                # And for bottom left point mass
                if (left_x, bottom_y) in point_masses:
                    bottom_left_index = point_masses.index((left_x, bottom_y))
                else:
                    bottom_left_index = len(point_masses)
                    point_masses.append((left_x, bottom_y))

                # And finally bottom right
                if (right_x, bottom_y) in point_masses:
                    bottom_right_index = point_masses.index((right_x, bottom_y))
                else:
                    bottom_right_index = len(point_masses)
                    point_masses.append((right_x, bottom_y))

                # Voxel types 3 and 4 are actuators. Dont' want to add voxel if its not an actuator
                if voxel_type in [3, 4]:
                    pmis = np.array([top_left_index, top_right_index, bottom_left_index, bottom_right_index])
                    actuator_obj = Actuator(actuator_action_index, voxel_type, pmis) 
                    actuators.append(actuator_obj)
                    actuator_action_index += 1

                left_x += 1

            top_y -= 1
            left_x = 0

        return actuators
    
    def get_corner_distances(self, pm_pos: list) -> list:
        """
        Given the list of robot point mass coordinates generated from sim.object_pos_at_time(),
        returns an list of an lists where each top level list corresponds to a an actuator voxel,
        the the sublist contains the distance to the [top left corner, top right corner,
        bottom left corner, bottom right corner].
        
        Parameters:
            pm_pos (list): A list with the first element being a np.ndarray containing all
                           point mass x positions, and second element containig all point mass
                           y positions.
        
        Returns:
            list: A list of an lists where each top level list corresponds to a an actuator voxel,
                  the the sublist contains the distance to the [top left corner, top right corner,
                  bottom left corner, bottom right corner].
        """

        actuator_distances = []

        for actuator in self.actuators:
            actuator_distances.append(actuator.get_distances_to_corners(pm_pos, self.corners))

        return actuator_distances

    def find_corners(self, robot_filepath):
        '''
        Finds the corners, returns them in an array of four corner objects.

        WARNING:
        This function expects a robot whose outermost corners are at the 
        minimum and maximum x and y values. 
        IT WILL BREAK IF THIS IS NOT THE CASE!!!

        Parameters:
            robot_file (string): name of the json file the robot lives in.

        Returns:
            An ndarray of four corner objects. Each represents an outer corner.
        '''

        env_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "world_data", ENV_FILENAME)

        world = EvoWorld.from_json(env_filepath)
        robot = WorldObject.from_json(robot_filepath)
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

        top_right = Corner(self.find_pm_index(xy_coords, x_max, y_max))
        top_left = Corner(self.find_pm_index(xy_coords, x_min, y_max))
        bottom_right = Corner(self.find_pm_index(xy_coords, x_max, y_min))
        bottom_left = Corner(self.find_pm_index(xy_coords, x_min, y_min))

        toreturn = np.array([top_right, top_left, bottom_right, bottom_left])

        return toreturn
    
    def find_pm_index(self, xy_coords, x_target, y_target):
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

