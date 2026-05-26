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
