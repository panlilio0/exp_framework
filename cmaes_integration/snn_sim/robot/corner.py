'''
A class representing corner of a robot in evogym.

Author: Matthew Meek
'''

import math
import numpy as np

class Corner:
    """
    A class reperesenting a corner of an evogym robot.
    Could represent any point-mass, really. But it was made for corners.

    ...

    Attributes
    ----------
    index : int:
        the index of the corner in it's pos array

    Methods
    -------
    get_corner_distances(positions, corners):
        Gets distances to the given corners in the given pos array and returns them.
    """

    def __init__(self, index):
        self.index = index # the index of the corner in the pos array.

    def __str__(self):
        return "Corner at " + str(self.index)
    
    def __repr__(self):
        return "corner("+str(self.index)+")"

    def get_corner_distances(self, positions, corners):
        '''
        Gets distances to the corners and returns them.

        Parameters:
            positions (ndarray): array containg all of the point mass coords
            corners (ndarray): array containing all of the corner objects

        Returns:
            An ndarray of floats with the distances from this corner to each other corner. 
        '''

        local_x = positions[0][self.index]
        local_y = positions[1][self.index]

        distances = []

        for corner in corners: # standard distance formula
            other_x = positions[0][corner.index]
            other_y = positions[1][corner.index]
            distance = math.sqrt((other_x-local_x)*(other_x-local_x) +
                                 (other_y-local_y)*(other_y-local_y))
            distances.append(distance)
        return np.array(distances)