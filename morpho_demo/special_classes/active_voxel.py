'''
A class representing an active voxel in evogym. 
'''

class active_voxel:

    def __init__(self, voxelindex, voxeltype):
        self.voxelindex = voxelindex # location of voxel in json grid (int)
        self.voxeltype = voxeltype # type of voxel as stored in json (int)

    def GDFOA(self):
        '''
        "Get Distance From Other Actives"
        Gets distance from other active voxels given x,y coords
        '''
        
