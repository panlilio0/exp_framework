"""
Representation of an evogym robot.

Authors: Thomas Breimer
February 21st, 2025
"""

import os
import numpy as np
from evogym import WorldObject
from actuator import Actuator

class Morpology:
    """
    Our own internal representation of an evogym robot.
    """

    def __init__(self, filename: str):
        """
        Given an evogym robot file, constructs a robot morphology.

        Parameters:
            filename (str): Filename of the robot .json file.
        """

        self.structure = self.get_structure(filename)
        self.actuators = self.create_actuator_voxels(self.structure)

    def get_structure(self, filename: str) -> np.ndarray:
        """
        Return the robot’s structure matrix.

        Parameters:
            filename (str): The filename of the robot to get.

        Returns:
            np.ndarray: (n, m) array specifing the voxel structure of the object.
        """

        # Get robot structure
        robot = WorldObject.from_json(os.path.join('world_data', filename))
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
                    actuators.push(actuator_obj)
                    actuator_action_index += 1

                left_x += 1

            top_y -= 1
            left_x = 0

        return actuators
