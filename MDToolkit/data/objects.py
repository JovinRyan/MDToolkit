import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from MDToolkit.paths import ELEMENTS_CSV
from MDToolkit.utils.structure_file_utils import create_elements_dictionary

class Topology:
  '''
  '''
  def __init__(self, type_mapping : dict, elements_dict : dict):
    self.bonds = None
    self.angles = None
    self.type_mapping = type_mapping
    self.elements_dict = elements_dict

class Frame:
  '''
  '''

  def __init__(self, topology : Topology):
    self.timestep = None
    self.box = None
    self.positions = None
    self.unwrapped_positions = None
    self.velocities = None
    self.forces = None
    self.topology = topology

class Reader(ABC):
  '''
  '''

  @abstractmethod
  def read_frame(self, idx: int) -> Frame:
    pass

  @abstractmethod
  def __iter__(self):
    pass

  @abstractmethod
  def __len__(self):
    pass


class Simulation:
  '''
  '''

  def __init__(self, filepath : Path, topology : Topology, reader : type[Reader]):
    self.reader = reader(filepath, topology)
    self.topology = topology
    self.filepath = filepath

  def __iter__(self):
    return iter(self.reader)

  def __getitem__(self, idx):
    return self.reader.read_frame(idx)

  def __len__(self):
    return len(self.reader)

class LAMMPS_CustomDump_Reader(Reader):
  '''
  '''

  def __init__(self, filepath : Path, topology : Topology):
    self.filepath = filepath
    self.topology = topology

    self._frame_offsets = []

    self._initialize()

  def read_frame(self, idx : int) -> Frame:
    pass

  def __iter__(self):
    for idx in range(len(self)):
      yield self.read_frame(idx)

  def __len__(self):
    return len(self._frame_offsets)

  def _initialize(self):
    with open(self.filepath, "r") as f:
      while True:
        position = f.tell()
        line = f.readline()

        if not line:
          break

        if line.startswith("ITEM: TIMESTEP"):
          self._frame_offsets.append(position)
