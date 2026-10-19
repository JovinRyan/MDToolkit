import os 
import numpy as np
from MDToolkit.data.objects import Frame
from MDToolkit.IO.write_file import write_lammps_data_file
from MDToolkit.paths import OUTPUT

def generate_windows(input_structure : Frame, atom_idx : int, positions):
    '''
    '''
