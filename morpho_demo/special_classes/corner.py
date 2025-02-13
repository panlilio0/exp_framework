'''
A class representing corner of a robot in evogym
'''

class corner:

    def __init__(self, index):
        self.index = index # the index of the corner in the 

    def __str__(self):
        return "Corner at: " + str(self.index)
    
    def __repr__(self):
        return "corner("+str(self.index)+")"

    def get_corner_distances(self, positions, corners):
        '''
        Desc

        Parameters:
            positions (ndarray): array containg all of the point mass coords
            corners (ndarray): array containing all of the corner objects

        Returns:
            An ndarray with the distances from this corner to each other corner. 
        '''