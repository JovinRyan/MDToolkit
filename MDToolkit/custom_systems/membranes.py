import math
import numpy as np 
from MDToolkit.data.objects import Topology, Frame
from MDToolkit.data.misc_objects import Volume
from MDToolkit.utils.misc_utils import rotation_matrix

def create_pore(frame : Frame, radius : float, position = [0, 0], radius_increment = 0.01, maintain_stoichiometry = True):
    '''
    '''

    if maintain_stoichiometry:
        frame.populate_stoich_dict()

        original_stoich = {
            i : frame.topology.stoich_dict[i]
            for i in frame.topology.stoich_dict
        }

        original_min = min(original_stoich.values())

        original_stoich = {
            i : original_stoich[i] / original_min
            for i in original_stoich
        }

    while True:

        mask = (
            (position[0] - frame.positions[:, 2]) ** 2
            + (position[1] - frame.positions[:, 1]) ** 2
        ) >= radius ** 2

        if maintain_stoichiometry:

            remaining_counts = {
                i : np.sum(frame.types[mask] == i)
                for i in frame.topology.stoich_dict
            }

            remaining_min = min(remaining_counts.values())

            remaining_stoich = {
                i : remaining_counts[i] / remaining_min
                for i in remaining_counts
            }

            if remaining_stoich == original_stoich:
                break

        else:
            break

        radius += radius_increment

    print(f"Pore radius: {radius:.2f} Å")

    if frame.ids is not None:
      frame.ids = frame.ids[mask]

    if frame.types is not None:
      frame.types = frame.types[mask]

    if frame.mol_ids is not None:
      frame.mol_ids = frame.mol_ids[mask]

    if frame.positions is not None:
      frame.positions = frame.positions[mask]

    if frame.unwrapped_positions is not None:
      frame.unwrapped_positions = frame.unwrapped_positions[mask]

    if frame.velocities is not None:
      frame.velocities = frame.velocities[mask]

    if frame.forces is not None:
      frame.forces = frame.forces[mask]

    frame.num_atoms = np.sum(mask)

    return frame

def create_2D_membrane(frame: Frame, dims = [50, 50], rotation_angles = [0, 90, 0]):
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

    frame.delete_atoms_outside_ranges(
        x_range = [0, math.ceil(dims[0] / original_box_lens[0]) * original_box_lens[0]],
        y_range = [0, math.ceil(dims[1] / original_box_lens[1]) * original_box_lens[1]],
        z_range = z_range
    )

    frame.rotate(rotation_angles)

    frame.set_center([0, 0, 0])

    return frame