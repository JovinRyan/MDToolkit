from MDToolkit.IO.read_file import cif_file_to_structured_system
from MDToolkit.data.objects import StructuredSystem
from MDToolkit.utils.cutom_systems_utils import get_min_periodic_image_number
from MDToolkit.utils.structure_file_utils import create_periodic_images, delete_atoms_outside_region


def cif_file_to_monolayer_membrane(unit_cell_file_path : str, max_dimension = [100, 100, 100], min_dimension = [0, 0, 0], rotation_angles = [90, 0, 0], membrane_thickness_buffer = 5) -> StructuredSystem:
    """
    Reads a CIF file, creates a structured system, generates periodic images to create a larger system, deletes atoms outside a specified region to isolate a monolayer membrane, rotates the system, and writes the resulting monolayer membrane to a PDB file.

    PARAMETERS:
    unit_cell_file_path (str): The path to the input CIF file.
    output_file_name (str): The name of the output PDB file.
    max_dimension (list of float): A list of three values representing the maximum coordinates in the x, y, and z dimensions, respectively. Default is [100, 100, 100].
    min_dimension (list of float): A list of three values representing the minimum coordinates in the x, y, and z dimensions, respectively. Default is [0, 0, 0].
    rotation_angles (list of float): A list of three angles (in degrees) for rotation around the x, y, and z axes, respectively. Default is [90, 0, 0].

    RETURNS:
    StructuredSystem: A structured system representing the monolayer membrane.
    """

    unit_cell_system = cif_file_to_structured_system(unit_cell_file_path)

    num_images = get_min_periodic_image_number(unit_cell_system, max_dimension[0] - min_dimension[0], max_dimension[1] - min_dimension[1], max_dimension[2] - min_dimension[2])
    num_images = (num_images[0], num_images[1], 2)

    extended_system = create_periodic_images(unit_cell_system, num_images)

    retention_region = {
        "min_z": (extended_system.box_dimensions["min_z"] + extended_system.box_dimensions["max_z"])/2 - membrane_thickness_buffer/2,
        "max_z": (extended_system.box_dimensions["min_z"] + extended_system.box_dimensions["max_z"])/2 + membrane_thickness_buffer/2,
        "min_y": extended_system.box_dimensions["min_y"],
        "max_y": extended_system.box_dimensions["max_y"],
        "min_x": extended_system.box_dimensions["min_x"],
        "max_x": extended_system.box_dimensions["max_x"]
    }

    monolayer_system = delete_atoms_outside_region(extended_system, retention_region)

    monolayer_system.set_all_molecules_id(1)

    monolayer_system.rotate_system_spherical(rotation_angles[0], rotation_angles[1], rotation_angles[2])

    return monolayer_system
