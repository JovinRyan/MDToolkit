import pandas as pd
import math
import numpy as np
from bidict import bidict
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from MDToolkit.utils.misc_utils import count_decimals
from MDToolkit.utils.structure_file_utils import create_elements_dictionary

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

  def __init__(self, id : int, element : str, position = None, charge : float = 0.0, chain_id : str = "A", type : str = "ATOM", elemental_properties = None, elemental_properties_keys = None):

    if position is None:
      position = [0.0, 0.0, 0.0]
    self.id = id
    self.element = element
    self.position = position
    self.charge = charge
    self.chain_id = chain_id
    self.type = type

    if elemental_properties is None:
      elemental_properties = {}
    self.elemental_properties = elemental_properties
    if elemental_properties_keys is None:
      elemental_properties_keys = []
    self.elemental_properties_keys = elemental_properties_keys

  def __repr__(self):
    return f"Atom(type='{self.type}', id={self.id}, element='{self.element}', position={self.position}, charge={self.charge}, chain_id='{self.chain_id}', elemental_properties='{self.elemental_properties}', elemental_properties_keys='{self.elemental_properties_keys}')"

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

  def fix_2char_element_symbol(self):
    '''
    Fixes the element symbol of this atom if it consists of two characters by capitalizing the first letter and making the second letter lowercase.\n
    INPUT:\n
    None

    RETURNS:\n
    None: This method modifies the element symbol of the atom in place and does not return anything.
    '''
    if len(self.element) == 2:
      self.element = self.element[0].upper() + self.element[1].lower()

  def populate_elemental_properties(self):
    '''
    Populates the elemental properties of this atom based on a provided dictionary mapping element symbols to their properties.\n
    INPUT:\n
    elemental_properties_dict (dict): A dictionary where keys are element symbols (e.g., 'H', 'O', 'C') and values are dictionaries containing properties for each element.

    RETURNS:\n
    None: This method modifies the elemental properties of the atom in place and does not return anything.
    '''

    try:
      elemental_properties_dict = create_elements_dictionary()[self.element]

      self.elemental_properties = elemental_properties_dict
      self.elemental_properties_keys = list(elemental_properties_dict.keys())
    except KeyError:
      self.fix_2char_element_symbol()
      elemental_properties_dict = create_elements_dictionary()[self.element]

      self.elemental_properties = elemental_properties_dict
      self.elemental_properties_keys = list(elemental_properties_dict.keys())

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
    self.bonds = []
    self.angles = []

  def __repr__(self):
    return f"Molecule(id={self.id}, name='{self.name}', atoms={self.atoms}, bonds={self.bonds}, angles={self.angles})"

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
    # self.rebase_coordinates_to_new_origin(new_origin=[0.0, 0.0, 0.0])

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
  
  def populate_bonds_from_molecular_data(self, molecular_data, bond_count = 1):
    '''
    '''
    bonds_list = []
    for i in range(len(molecular_data["bond_atom_indices"])):
      bonds_list.append([i + bond_count, molecular_data["bond_types"][i], self.atoms[molecular_data["bond_atom_indices"][i][0]].id, self.atoms[molecular_data["bond_atom_indices"][i][1]].id])
    
    self.bonds = bonds_list

  def populate_angles_from_molecular_data(self, molecular_data, angle_count = 1):
    '''
    '''
    angles_list = []
    for i in range(len(molecular_data["angle_bond_indices"])):
      angles_list.append([i + angle_count, molecular_data["angle_types"][i], self.atoms[molecular_data["angle_bond_indices"][i][0]].id, self.atoms[molecular_data["angle_bond_indices"][i][1]].id, self.atoms[molecular_data["angle_bond_indices"][i][2]].id])

    self.angles = angles_list

class StructuredSystem:
  '''
  Represents a structured molecular system.\n
  Parameters:
- molecule_list (list): A list of Molecule objects that make up the structured system.
- box_dimensions (dict): A dictionary containing the minimum and maximum coordinates of the system in each dimension (e.g., {"min_x": 0.0, "max_x": 10
.0, "min_y": 0.0, "max_y": 10.0, "min_z": 0.0, "max_z": 10.0}). Default is None.
- box_angles (dict): A dictionary containing the angles between the box vectors (e.g., {"angle_ab": 90.0, "angle_ac": 90.0, "angle_bc": 90.0}). Default is None.\n
Methods:
- __init__(self, molecule_list, box_dimensions = None, box_angles = None): Initializes a StructuredSystem object with the given parameters.
- __repr__(self): Returns a string representation of the StructuredSystem object.
- combine_with_other_structured_system(self, other_system): Combines this structured system with another structured system by merging their molecule lists and updating box dimensions and angles accordingly.
- set_all_molecules_id(self, new_id = 1): Sets the same ID for all molecules in the structured system.
- rotate_system_spherical(self, theta_deg, phi_deg, psi_deg): Rotates the entire structured system using spherical coordinates via ZYZ Euler angles.
- populate_elemental_properties_for_all_atoms(self): Populates the elemental properties for all atoms in all molecules of the structured system based on a provided dictionary mapping element symbols to their properties.
- check_if_all_atoms_have_elemental_properties(self): Checks if all atoms in all molecules of the structured system have their elemental properties populated.
  '''
  def __init__(self, molecule_list, box_dimensions = None, box_angles = None):
    self.molecule_list = molecule_list
    self.box_dimensions = box_dimensions
    if box_angles is None:
      box_angles = [90.0, 90.0, 90.0]
    self.box_angles = box_angles
    self._elemental_properties_populated = False

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

  def set_all_molecules_id(self, new_id = 1):
    '''
    Sets the same ID for all molecules in the structured system.\n
    INPUT:\n
    new_id (int): The new ID to be assigned to all molecules in the structured system. Default is 1.

    RETURNS:\n
    None: This method modifies the molecule IDs in place and does not return anything.
    '''
    for molecule in self.molecule_list:
      molecule.id = new_id

  def reset_atom_ids(self):
    new_id = 1
    for molecule in self.molecule_list:
      for atom in molecule.atoms:
        atom.id = new_id
        new_id += 1

  def reset_molecule_ids(self):
    new_id = 1
    for molecule in self.molecule_list:
      molecule.id = new_id
      new_id += 1

  def rotate_system_spherical(self, theta_deg, phi_deg, psi_deg):
    '''
    Rotates the entire structured system using spherical coordinates via ZYZ Euler angles.\n
    INPUT:\n
    theta_deg (float): Polar angle in degrees — tilt from the z-axis (0 to 180).\n
    phi_deg (float): Azimuthal angle in degrees — rotation in the xy-plane (0 to 360).\n
    psi_deg (float): Roll angle in degrees — rotation around the radial vector (0 to 360).\n
    RETURNS:\n
    None: This method modifies the positions of the atoms in place and does not return anything.
    '''
    all_atoms = [atom for molecule in self.molecule_list for atom in molecule.atoms]

    positions = np.array([atom.position for atom in all_atoms])

    rotation = R.from_euler('ZYZ', [phi_deg, theta_deg, psi_deg], degrees=True)

    rotated_positions = rotation.apply(positions)

    for i in range(len(rotated_positions)):
      all_atoms[i].position = rotated_positions[i]

    vertices = np.array([
      [self.box_dimensions["min_x"], self.box_dimensions["min_y"], self.box_dimensions["min_z"]],
      [self.box_dimensions["max_x"], self.box_dimensions["min_y"], self.box_dimensions["min_z"]],
      [self.box_dimensions["min_x"], self.box_dimensions["max_y"], self.box_dimensions["min_z"]],
      [self.box_dimensions["min_x"], self.box_dimensions["min_y"], self.box_dimensions["max_z"]],
      [self.box_dimensions["max_x"], self.box_dimensions["max_y"], self.box_dimensions["min_z"]],
      [self.box_dimensions["max_x"], self.box_dimensions["min_y"], self.box_dimensions["max_z"]],
      [self.box_dimensions["min_x"], self.box_dimensions["max_y"], self.box_dimensions["max_z"]],
      [self.box_dimensions["max_x"], self.box_dimensions["max_y"], self.box_dimensions["max_z"]],
      ])

    rotated_vertices = rotation.apply(vertices).tolist()
    vertices_x_coordinates = [vertex[0] for vertex in rotated_vertices]
    vertices_y_coordinates = [vertex[1] for vertex in rotated_vertices]
    vertices_z_coordinates = [vertex[2] for vertex in rotated_vertices]

    self.box_dimensions["min_x"] = min(vertices_x_coordinates)
    self.box_dimensions["max_x"] = max(vertices_x_coordinates)
    self.box_dimensions["min_y"] = min(vertices_y_coordinates)
    self.box_dimensions["max_y"] = max(vertices_y_coordinates)
    self.box_dimensions["min_z"] = min(vertices_z_coordinates)
    self.box_dimensions["max_z"] = max(vertices_z_coordinates)

  def populate_elemental_properties_for_all_atoms(self, properties = ("AtomicMass")):
    '''
    Populates the elemental properties for all atoms in all molecules of the structured system based on a provided dictionary mapping element symbols to their properties.\n
    INPUT:\n
    None

    RETURNS:\n
    None: This method modifies the elemental properties of the atoms in place and does not return anything.
    '''

    elements_dict = create_elements_dictionary()

    if isinstance(properties, str):
        properties = (properties,)

    elemental_property_cache = {}
    elemental_keys_cache = {}

    for molecule in self.molecule_list:
        for atom in molecule.atoms:

            element = atom.element

            if element not in elemental_property_cache:
                try:
                    full_props = elements_dict[element]

                except KeyError:
                    atom.fix_2char_element_symbol()
                    element = atom.element
                    full_props = elements_dict[element]

                filtered_props = {
                    key: full_props[key]
                    for key in properties
                    if key in full_props
                }

                elemental_property_cache[element] = filtered_props
                elemental_keys_cache[element] = tuple(filtered_props.keys())

            atom.elemental_properties = elemental_property_cache[element]
            atom.elemental_properties_keys = elemental_keys_cache[element]

    self._elemental_properties_populated = True

  def check_if_all_atoms_have_elemental_properties(self):
    '''
    Checks if all atoms in all molecules of the structured system have their elemental properties populated.\n
    INPUT:\n
    None

    RETURNS:\n
    bool: True if all atoms have their elemental properties populated, False otherwise.
    '''
    return self._elemental_properties_populated

  def find_COM(self):
    if not self.check_if_all_atoms_have_elemental_properties():
        self.populate_elemental_properties_for_all_atoms()

    mass_cache = {}

    all_atoms = [atom for molecule in self.molecule_list for atom in molecule.atoms]

    positions = np.array([atom.position for atom in all_atoms])
    masses = np.array([
        mass_cache.setdefault(atom.element, atom.elemental_properties["AtomicMass"]) for atom in all_atoms
    ])

    center_of_mass = np.dot(masses, positions) / np.sum(masses)

    return center_of_mass.tolist()

  def set_COM_to_origin(self, origin = [0, 0, 0]):
    COM = self.find_COM()

    for molecule in self.molecule_list:
        for atom in molecule.atoms:
            atom.position = (np.array(atom.position) - np.array(COM) + np.array(origin)).tolist()

    width_x = self.box_dimensions["max_x"] - self.box_dimensions["min_x"]
    width_y = self.box_dimensions["max_y"] - self.box_dimensions["min_y"]
    width_z = self.box_dimensions["max_z"] - self.box_dimensions["min_z"]

    self.box_dimensions["min_x"] = origin[0] - width_x / 2
    self.box_dimensions["max_x"] = origin[0] + width_x / 2

    self.box_dimensions["min_y"] = origin[1] - width_y / 2
    self.box_dimensions["max_y"] = origin[1] + width_y / 2

    self.box_dimensions["min_z"] = origin[2] - width_z / 2
    self.box_dimensions["max_z"] = origin[2] + width_z / 2

  def get_system_stoich_dict(self):
    all_atoms_element_list = np.array([atom.element for molecule in self.molecule_list for atom in molecule.atoms])
    elements_list, counts = np.unique(all_atoms_element_list, return_counts=True)
    stoich_list = [count/min(counts) for count in counts]

    return dict(zip(elements_list, stoich_list))

  def build_nD_tree(self, axes : list = ["x", "y", "z"]):
    axes_indices = []

    if "x" in axes:
      axes_indices.append(0)
    if "y" in axes:
      axes_indices.append(1)
    if "z" in axes:
      axes_indices.append(2)

    axes_indices.sort()

    positions = np.array([[atom.position[i] for i in axes_indices] for molecule in self.molecule_list for atom in molecule.atoms])

    return cKDTree(positions)

  def delete_atoms_by_ids(self, atom_ids, resset_ids = True):
    '''
    '''

    atom_ids = set(atom_ids)
    new_molecule_list = []

    for molecule in self.molecule_list:
      molecule.atoms = [atom for atom in molecule.atoms if atom.id not in atom_ids]

      if len(molecule.atoms) > 0:
        new_molecule_list.append(molecule)

    self.molecule_list = new_molecule_list

    if resset_ids:
      self.reset_atom_ids()
      self.reset_molecule_ids()

  def set_box_dimensions_to_vdw(self):
    if not self.check_if_all_atoms_have_elemental_properties():
        self.populate_elemental_properties_for_all_atoms()

    positions = np.array([
        atom.position
        for molecule in self.molecule_list
        for atom in molecule.atoms
    ])

    radii = np.array([
        atom.elemental_properties["AtomicRadius"]/100
        for molecule in self.molecule_list
        for atom in molecule.atoms
    ])[:, np.newaxis]

    min_vdw_coords = positions - radii
    max_vdw_coords = positions + radii

    self.box_dimensions["min_x"] = np.min(min_vdw_coords[:, 0])
    self.box_dimensions["max_x"] = np.max(max_vdw_coords[:, 0])

    self.box_dimensions["min_y"] = np.min(min_vdw_coords[:, 1])
    self.box_dimensions["max_y"] = np.max(max_vdw_coords[:, 1])

    self.box_dimensions["min_z"] = np.min(min_vdw_coords[:, 2])
    self.box_dimensions["max_z"] = np.max(max_vdw_coords[:, 2])

  def get_box_lengths(self):
    return self.box_dimensions["max_x"] - self.box_dimensions["min_x"], self.box_dimensions["max_y"] - self.box_dimensions["min_y"], self.box_dimensions["max_z"] - self.box_dimensions["min_z"]

  def populate_atom_charges(self, charge_dict: dict):
      get_charge = charge_dict.get

      for molecule in self.molecule_list:
          for atom in molecule.atoms:
              charge = get_charge(atom.element)
              if charge is not None:
                  atom.charge = charge

  def get_bond_counts(self):
    '''
    Returns:
        tuple: (number_of_bonds, number_of_bond_types)
    '''

    num_bonds = 0
    bond_types = set()

    for molecule in self.molecule_list:
        num_bonds += len(molecule.bonds)

        for bond in molecule.bonds:
            bond_types.add(bond[1])

    return num_bonds, len(bond_types)
  
  def get_angle_counts(self):
    '''
    Returns:
        tuple: (number_of_angles, number_of_angle_types)
    '''

    num_angles = 0
    angle_types = set()

    for molecule in self.molecule_list:
      num_angles += len(molecule.angles)

      for angle in molecule.angles:
        angle_types.add(angle[1])
    
    return num_angles, len(angle_types)
  
  def reset_bond_ids(self):
    '''
    '''
    bond_id = 1

    for molecule in self.molecule_list:
      for bond in molecule.bonds:
        bond[0] = bond_id 
        bond_id += 1

  def reset_angle_ids(self):
    '''
    '''
    angle_id = 1

    for molecule in self.molecule_list:
      for angle in molecule.angles:
        angle[0] = angle_id 
        angle_id += 1
  
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
        atom.fix_2char_element_symbol()
        atoms.append(atom)
      molecule = Molecule(
        molecule_id=molecule_index,
        molecule_name=molecule_df["molecule_name"].iloc[0],
        atoms=atoms
      )
      molecule_list.append(molecule)

    return molecule_list

class Simulation:
  '''
  Represents a molecular dynamics simulation.\n
  Parameters:
- frames (list): A list of StructuredSystem objects representing the frames of the simulation.
- timesteps (list): A list of timesteps corresponding to each frame in the simulation.
- atom_counts (list): A list of atom counts corresponding to each frame in the simulation.\n
Methods:
- __init__(self, structured_systems_list, timesteps_list, atom_count_list):
  '''
  def __init__(self, structured_systems_list, timesteps_list, atom_count_list):
    self.frames = structured_systems_list
    self.timesteps = timesteps_list
    self.atom_counts = atom_count_list
    self._elemental_properties_populated = False
  
  def populate_elemental_properties_for_all_frames(self, properties = ("AtomicMass")):
    for frame in self.frames:
      frame.populate_elemental_properties_for_all_atoms(properties)
    
    self._elemental_properties_populated = True