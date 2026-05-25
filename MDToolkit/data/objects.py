import pandas as pd
import math
from MDToolkit.utils.misc_utils import count_decimals

class Atom:
  '''
  Represents an atom in the molecular system.\n
  Parameters:
- type (str): The type of the atom (e.g., "ATOM" or "HETATM"). Default is "ATOM".
- id (int): Unique identifier for the atom.
- element (str): Chemical element symbol (e.g., 'H', 'O', 'C').
- position (list): A list of three floats representing the x, y, z coordinates of the atom. Default is [0.0, 0.0, 0.0].
- charge (float): The charge of the atom. Default is 0.0.
- chain_id (str): The chain identifier for the atom. Default is "A".\n
Methods:
- __init__(self, id: int, element: str, position=None, charge: float = 0.0, chain_id: str = "A"): Initializes an Atom object with the given parameters.
- __repr__(self): Returns a string representation of the Atom object.
  '''

  def __init__(self, id : int, element : str, position = None, charge : float = 0.0, chain_id : str = "A", type : str = "ATOM"):

    if position is None:
      position = [0.0, 0.0, 0.0]
    self.id = id
    self.element = element
    self.position = position
    self.charge = charge
    self.chain_id = chain_id
    self.type = type

  def __repr__(self):
    return f"Atom(type='{self.type}', id={self.id}, element='{self.element}', position={self.position}, charge={self.charge}, chain_id='{self.chain_id}' ')"

  def get_distance_to(self, coordinate):
    '''
    Calculates the distance from this atom to a point in 3D space defined by the input coordinates.\n
    INPUT:\n
    3_D_coordinates (list): A list of three floats representing the x, y, z coordinates of the point to which the distance is calculated.

    RETURNS:\n
    distance (float): The Euclidean distance from this atom to the specified point in 3D space.
    '''

    dist = math.dist(self.position, coordinate)
    return dist

class Molecule:
  '''
  Represents a molecule in the molecular system.\n
  Parameters:
- molecule_id (int): Unique identifier for the molecule.
- molecule_name (str): Name of the molecule.
- atoms (list): A list of Atom objects that make up the molecule. Default is an empty list.\n
Methods:
- __init__(self, molecule_id : int, molecule_name : str, atoms = None): Initializes a Molecule object with the given parameters.
- __repr__(self): Returns a string representation of the Molecule object.
  '''

  def __init__(self, molecule_id : int, molecule_name : str, atoms = None):

    if atoms is None:
      atoms = []
    self.id = molecule_id
    self.name = molecule_name
    self.atoms = atoms

  def __repr__(self):
    return f"Molecule(id={self.id}, name='{self.name}', atoms={self.atoms})"

  def get_all_distances_to(self, coordinate):
    '''
    Calculates the distances from all atoms in this molecule to a point in 3D space defined by the input coordinates.\n
    INPUT:\n
    3_D_coordinates (list): A list of three floats representing the x, y, z coordinates of the point to which the distances are calculated.

    RETURNS:\n
    distances (list): A list of floats representing the Euclidean distances from each atom in this molecule to the specified point in 3D space.
    '''

    distances = [atom.get_distance_to(coordinate) for atom in self.atoms]
    return distances

  def rebase_coordinates_to_new_origin(self, new_origin = [0.0, 0.0, 0.0]):
    '''
    Rebase the coordinates of all atoms in this molecule to a new origin defined by the input coordinates.\n
    INPUT:\n
    new_origin (list): A list of three floats representing the x, y, z coordinates of the new origin to which the atom coordinates will be rebased.

    RETURNS:\n
    None: This method modifies the positions of the atoms in place and does not return anything.
    '''

    max_atom_coords_decimals = max(count_decimals(atom.position[i]) for atom in self.atoms for i in range(3))

    distances_to_new_origin = self.get_all_distances_to(new_origin)
    min_index = distances_to_new_origin.index(min(distances_to_new_origin))
    closest_atom = self.atoms[min_index]
    offset_vector = [closest_atom.position[i] - new_origin[i] for i in range(3)]
    for atom in self.atoms:
      atom.position = [round(atom.position[i] - offset_vector[i], max_atom_coords_decimals) for i in range(3)]

  def rotate_molecule_spherical(self, theta_deg, phi_deg, psi_deg):
    '''
    Rotates the molecule using spherical coordinates via ZYZ Euler angles.\n
    INPUT:\n
    theta_deg (float): Polar angle in degrees — tilt from the z-axis (0 to 180).\n
    phi_deg (float): Azimuthal angle in degrees — rotation in the xy-plane (0 to 360).\n
    psi_deg (float): Roll angle in degrees — rotation around the radial vector (0 to 360).\n
    RETURNS:\n
    None: This method modifies the positions of the atoms in place and does not return anything.
    '''
    self.rebase_coordinates_to_new_origin(new_origin=[0.0, 0.0, 0.0])

    phi   = math.radians(phi_deg)
    theta = math.radians(theta_deg)
    psi   = math.radians(psi_deg)

    # 1. Rotate around z by phi (azimuthal)
    R_z1 = [[math.cos(phi), -math.sin(phi), 0],
            [math.sin(phi),  math.cos(phi), 0],
            [0,              0,             1]]

    # 2. Rotate around y by theta (polar tilt)
    R_y  = [[ math.cos(theta), 0, math.sin(theta)],
            [ 0,               1, 0              ],
            [-math.sin(theta), 0, math.cos(theta)]]

    # 3. Rotate around z again by psi (roll around radial vector)
    R_z2 = [[math.cos(psi), -math.sin(psi), 0],
            [math.sin(psi),  math.cos(psi), 0],
            [0,              0,             1]]

    # Combined: R = R_z1 @ R_y @ R_z2
    R_zy  = [[sum(R_z1[i][k] * R_y[k][j]  for k in range(3)) for j in range(3)] for i in range(3)]
    R     = [[sum(R_zy[i][k] * R_z2[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    for atom in self.atoms:
        x_new = R[0][0]*atom.position[0] + R[0][1]*atom.position[1] + R[0][2]*atom.position[2]
        y_new = R[1][0]*atom.position[0] + R[1][1]*atom.position[1] + R[1][2]*atom.position[2]
        z_new = R[2][0]*atom.position[0] + R[2][1]*atom.position[1] + R[2][2]*atom.position[2]
        atom.position = [x_new, y_new, z_new]

  def displace_to_non_negative_coordinates(self):
    '''
    Displaces the molecule so that all atom coordinates are non-negative.\n
    INPUT:\n
    None

    RETURNS:\n
    None: This method modifies the positions of the atoms in place and does not return anything.
    '''
    min_coords = [min(atom.position[i] for atom in self.atoms) for i in range(3)]
    displacement_vector = [-min_coords[i] if min_coords[i] < 0 else 0 for i in range(3)]
    for atom in self.atoms:
      atom.position = [atom.position[i] + displacement_vector[i] for i in range(3)]


class StructuredSystem:
  '''
  Represents a structured molecular system.\n
  Parameters:
  '''
  def __init__(self, molecule_list, box_dimensions = None, box_angles = None):
    self.molecule_list = molecule_list
    self.box_dimensions = box_dimensions
    if box_angles is None:
      box_angles = [90.0, 90.0, 90.0]
    self.box_angles = box_angles

  def __repr__(self):
    return f"StructuredSystem(molecule_list={self.molecule_list}, box_dimensions={self.box_dimensions}, box_angles={self.box_angles})"

  def combine_with_other_structured_system(self, other_system):
    '''
    Combines this structured system with another structured system by merging their molecule lists and updating box dimensions and angles accordingly.\n
    INPUT:\n
    other_system (StructuredSystem): Another StructuredSystem object to be combined with this one.

    RETURNS:\n
    None: This method modifies the current StructuredSystem in place by combining it with the other system and does not return anything.
    '''

    system_max_molecule_index = max(molecule.id for molecule in self.molecule_list)
    system_max_atom_index = max(atom.id for molecule in self.molecule_list for atom in molecule.atoms)
    for molecule in other_system.molecule_list:
      molecule.id += system_max_molecule_index
      for atom in molecule.atoms:
        atom.id += system_max_atom_index

    self.molecule_list = self.molecule_list + other_system.molecule_list

    self.box_dimensions = {
        "min_x": min(self.box_dimensions["min_x"], other_system.box_dimensions["min_x"]),
        "max_x": max(self.box_dimensions["max_x"], other_system.box_dimensions["max_x"]),
        "min_y": min(self.box_dimensions["min_y"], other_system.box_dimensions["min_y"]),
        "max_y": max(self.box_dimensions["max_y"], other_system.box_dimensions["max_y"]),
        "min_z": min(self.box_dimensions["min_z"], other_system.box_dimensions["min_z"]),
        "max_z": max(self.box_dimensions["max_z"], other_system.box_dimensions["max_z"])
    }


def construct_molecule_list_from_df(system_df):
    '''
    Constructs a list of Molecule objects from a pandas DataFrame containing molecular system information.
    INPUT:\n
    system_df (pandas DataFrame): A DataFrame containing columns such as "atom_index", "atom_species", "molecule_name", "chain_id", "molecule_index", "x", "y", "z", etc., which describe the atoms and their properties in the molecular system.

    RETURNS:\n
    molecule_list (list): A list of Molecule objects constructed from the input DataFrame.
    '''

    molecule_list = []

    grouped_system_df = system_df.groupby("molecule_index")

    for molecule_index, molecule_df in grouped_system_df:
      atoms = []
      for _, atom_row in molecule_df.iterrows():
        atom = Atom(
          type=atom_row["atom_type"],
          id=atom_row["atom_index"],
          element=atom_row["atom_species"],
          position=[atom_row["x"], atom_row["y"], atom_row["z"]],
          charge=0.0,
          chain_id=atom_row["chain_id"]
        )
        atoms.append(atom)
      molecule = Molecule(
        molecule_id=molecule_index,
        molecule_name=molecule_df["molecule_name"].iloc[0],
        atoms=atoms
      )
      molecule_list.append(molecule)

    return molecule_list

