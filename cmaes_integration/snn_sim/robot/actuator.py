'''
A class representing an active voxel in evogym. 

Author: Matthew Meek
'''

import math
import numpy as np

class Actuator:
    """
    A class reperesenting an actuator of an evogym robot.
    Could represent any voxel, really. But it was made for corners.
    ...

    Attributes
    ----------
    voxelindexA : int:
        index of the voxel in the action array.

    voxeltype : int:
        type of voxel in evogym json file.

    pmis : ndarray(int):
        indicies of the voxel's point-masses in the pos array.
        [top_left, top_right, bottom_left, bottom_right]

    Methods
    -------
    get_center_of_mass(self, positions):
        Gets xy-coords of the voxel's center of mass.

    get_distances_to_corners(self, positions, corners):
        Gets distances to the given corners in the given pos array and returns them.
    """

    def __init__(self, voxel_index_a, voxeltype, pmis):
        self.voxel_index_a = voxel_index_a # location of voxel in action array (int)
        self.voxeltype = voxeltype # type of voxel as stored in json (int)
        self.pmis = pmis # pos_array indicies of voxel point-masses (ndarray of int)

    def get_center_of_mass(self, positions):
        '''
        Gets x and y coords of the voxel's center of mass.

        Parameters:
            positions (ndarray): array containg all of the point mass coords

        Returns:
            A pair of ints. The x and y coords of the voxel's center of mass. 
        '''

        x_locs = [positions[0][self.pmis[0]],
                  positions[0][self.pmis[1]],
                  positions[0][self.pmis[2]],
                  positions[0][self.pmis[3]]]

        y_locs = [positions[1][self.pmis[0]],
                  positions[1][self.pmis[1]],
                  positions[1][self.pmis[2]],
                  positions[1][self.pmis[3]]]

        x_center = sum(x_locs) / 4
        y_center = sum(y_locs) / 4

        return x_center, y_center

    def get_distances_to_corners(self, positions, corners):
        '''
        Gets distances to the corners and returns them.

        Parameters:
            positions (ndarray): array containg all of the point mass coords
            corners (ndarray): array containing all of the corner objects

        Returns:
            An list of floats with the distances to each corner. 
        '''

        local_x, local_y = self.get_center_of_mass(positions)

        distances = []

        for corner in corners:
            other_x = positions[0][corner.index]
            other_y = positions[1][corner.index]
            distance = math.sqrt((other_x-local_x)*(other_x-local_x) +
                                 (other_y-local_y)*(other_y-local_y))
            distances.append(distance)
        return distances
