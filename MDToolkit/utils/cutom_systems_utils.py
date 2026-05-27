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

def find_center_of_mass(system : StructuredSystem):
    """
    Calculates the center of mass of a structured system.

    PARAMETERS:\n
    system (StructuredSystem): The structured system for which to calculate the center of mass.

    RETURNS:\n
    list: A list of three floats representing the x, y, and z coordinates of the center of mass, respectively.
    """

    if not system.check_if_all_atoms_have_elemental_properties():
        system.populate_elemental_properties_for_all_atoms()

    mass_cache = {}

    all_atoms = [atom for molecule in system.molecule_list for atom in molecule.atoms]

    positions = np.array([atom.position for atom in all_atoms])
    masses = np.array([
        mass_cache.setdefault(atom.element, atom.elemental_properties["AtomicMass"]) for atom in all_atoms
    ])

    center_of_mass = np.dot(masses, positions) / np.sum(masses)

    return center_of_mass.tolist()


