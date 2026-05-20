import pandas as pd

class Atom:
  '''
  Represents an atom in the molecular system.\n
  Parameters:
- id (int): Unique identifier for the atom.
- element (str): Chemical element symbol (e.g., 'H', 'O', 'C').
- position (list): A list of three floats representing the x, y, z coordinates of the atom. Default is [0.0, 0.0, 0.0].
- charge (float): The charge of the atom. Default is 0.0.
- chain_id (str): The chain identifier for the atom. Default is "A".\n
Methods:
- __init__(self, id: int, element: str, position=None, charge: float = 0.0, chain_id: str = "A"): Initializes an Atom object with the given parameters.
- __repr__(self): Returns a string representation of the Atom object.
  '''

  def __init__(self, id : int, element : str, position = None, charge : float = 0.0, chain_id : str = "A"):

    if position is None:
      position = [0.0, 0.0, 0.0]
    self.id = id
    self.element = element
    self.position = position
    self.charge = charge
    self.chain_id = chain_id

  def __repr__(self):
    return f"Atom(id={self.id}, element='{self.element}', position={self.position}, charge={self.charge}, chain_id='{self.chain_id}')"

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

class StructuredSystem:
  '''
  Represents a structured molecular system.\n
  Parameters:
  '''


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

