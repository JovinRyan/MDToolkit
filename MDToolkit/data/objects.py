class Atom:
  '''
  '''

  def __init__(self, id : int, element : str, position : list = [0.0, 0.0, 0.0]):
    self.id = id
    self.element = element
    self.position = position

class Molecule:
  '''
  '''

  def __init__(self, molecule_id : int, molecule_name : str, atoms : list = []):
    self.id = molecule_id
    self.name = molecule_name
    self.atoms = atoms
