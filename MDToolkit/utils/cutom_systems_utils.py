import numpy as np
from MDToolkit.data.objects import *

def get_min_periodic_image_number(unit_cell_system : StructuredSystem, max_x : float, max_y : float, max_z : float):
    """
    Calculates the minimum number of periodic images required in each dimension to cover a specified maximum distance.

    PARAMETERS:\n
    unit_cell_system (StructuredSystem): The structured system representing the unit cell.\n
    max_x (float): The maximum distance in the x-dimension that needs to be covered.\n
    max_y (float): The maximum distance in the y-dimension that needs to be covered.\n
    max_z (float): The maximum distance in the z-dimension that needs to be covered.\n

    RETURNS:\n
    tuple: A tuple containing three integers representing the number of periodic images needed in the x, y, and z dimensions, respectively.
    """
    a_length = unit_cell_system.box_dimensions["max_x"] - unit_cell_system.box_dimensions["min_x"]
    b_length = unit_cell_system.box_dimensions["max_y"] - unit_cell_system.box_dimensions["min_y"]
    c_length = unit_cell_system.box_dimensions["max_z"] - unit_cell_system.box_dimensions["min_z"]

    num_images_x = int(max_x / a_length) + 1
    num_images_y = int(max_y / b_length) + 1
    num_images_z = int(max_z / c_length) + 1

    return (num_images_x, num_images_y, num_images_z)

def create_pore_circular(structured_system, radius=10.0, origin=[0, 0], min_radius=1.0, maintain_stoich=True, radius_step=0.5, tolerance=1e-5):

    atom_list = np.array([
        atom
        for molecule in structured_system.molecule_list
        for atom in molecule.atoms
    ])

    yz_positions = np.array([
        [atom.position[1], atom.position[2]]
        for atom in atom_list
    ])

    origin = np.array(origin)

    # Distance from pore center
    distances = np.linalg.norm(
        yz_positions - origin,
        axis=1
    )

    # Reference stoich
    if maintain_stoich:
        stoich_dict = structured_system.get_system_stoich_dict()

    current_radius = radius

    while current_radius >= min_radius:

        mask = distances <= current_radius

        selected_atoms = atom_list[mask]

        if not maintain_stoich:
            break

        selected_elements = np.array([
            atom.element for atom in selected_atoms
        ])

        elements, counts = np.unique(
            selected_elements,
            return_counts=True
        )

        if len(counts) == 0:
            current_radius -= radius_step
            continue

        min_count = counts.min()

        selected_stoich_dict = {
            elem: count / min_count
            for elem, count in zip(elements, counts)
        }

        stoich_match = True

        for elem, ref_value in stoich_dict.items():

            if elem not in selected_stoich_dict:
                stoich_match = False
                break

            if not np.isclose(
                selected_stoich_dict[elem],
                ref_value,
                atol=tolerance
            ):
                stoich_match = False
                break

        if stoich_match:
            break

        current_radius -= radius_step

    atoms_to_delete = [
        atom.id for atom in selected_atoms
    ]

    structured_system.delete_atoms_by_ids(atoms_to_delete)

    return structured_system, np.pi * current_radius**2


def get_cif_file_cell_lengths(file_path):
    '''
    '''

    with open(file_path, 'r') as f:
        lines = f.readlines()

        for i in range(len(lines)):
            match lines[i].strip().split()[0]:
                case "_cell_length_a" : len_a = float(lines[i].strip().split()[1])
                case "_cell_length_b" : len_b = float(lines[i].strip().split()[1])
                case "_cell_length_c" : len_c = float(lines[i].strip().split()[1])

    return len_a, len_b, len_c



