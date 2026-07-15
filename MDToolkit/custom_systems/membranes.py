import math
import numpy as np 
from MDToolkit.data.objects import Topology, Frame
from MDToolkit.data.misc_objects import Volume
from MDToolkit.utils.misc_utils import rotation_matrix

def create_pore(frame : Frame, radius : float, position = [0, 0], maintain_stoichiometry = True):
    '''
    '''

def create_2D_membrane(frame: Frame, dims = [50, 50]):
    '''
    '''
    dims = np.array(dims)

    original_box_lens = frame.box.central_box_lengths

    x_images, y_images = np.ceil(dims / original_box_lens[:2]).astype(int)

    frame = frame.replicate(images = [x_images, y_images, 2])

    box_lens = frame.box.central_box_lengths

    z_range = [
        frame.box.origin[2] + (box_lens[2] / 4),
        frame.box.origin[2] + (box_lens[2] * 3 / 4)
    ]

    print(z_range)

    frame.delete_atoms_outside_ranges(
        x_range = [0, math.ceil(dims[0] / original_box_lens[0]) * original_box_lens[0]],
        y_range = [0, math.ceil(dims[1] / original_box_lens[1]) * original_box_lens[1]],
        z_range = z_range
    )

    frame.rotate([90, 0, 0])

    return frame