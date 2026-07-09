import os
import numpy as np
from pathlib import Path
from abc import ABC, abstractmethod
from MDToolkit.paths import ELEMENTS_CSV
from MDToolkit.utils.structure_file_utils import create_elements_dictionary
from MDToolkit.data.misc_objects import BoxVolume

class Topology:
  '''
  '''
  def __init__(self, type_mapping : dict, elements_dict : dict, charges_dict : dict = None):
    self.bonds = None
    self.angles = None
    self.type_mapping = type_mapping
    self.elements_dict = elements_dict
    self.charges = {key : 0.0 for key in type_mapping.keys()}
    if charges_dict is not None:
      for key in charges_dict.keys():
        self.charges[key] = charges_dict[key]

class Frame:
  '''
  '''

  def __init__(self, topology : Topology):
    self.timestep = None
    self.box = None
    self.ids = None 
    self.types = None
    self.mol_ids = None
    self.positions = None
    self.unwrapped_positions = None
    self.velocities = None
    self.forces = None
    self.topology = topology
    self.num_atoms = None
  
  def get_charges(self) -> np.ndarray:

    charges = np.empty(self.num_atoms, dtype = np.float64)

    for i, atom_type in enumerate(self.types):
      charges[i] = self.topology.charges[atom_type]
    
    return charges
  
  def get_masses(self) -> np.ndarray:

    masses = np.empty(self.num_atoms, dtype = np.float64)

    for i, atom_type in enumerate(self.types):
      element = self.topology.type_mapping[atom_type]
      masses[i] = self.topology.elements_dict[element]["AtomicMass"]
    
    return masses

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
  
  def close(self):
    self.reader.close()

class LAMMPS_CustomDump_Reader(Reader):
  '''
  '''

  def __init__(self, filepath : Path, topology : Topology):
    self.filepath = filepath
    self.topology = topology

    self._file = open(filepath, "rb")
    self._file.seek(0, os.SEEK_END)
    self._filesize = self._file.tell()
    self._file.seek(0)

    self._frame_offsets = []

    self._initialize()

  def read_frame(self, idx : int) -> Frame: 
    self._file.seek(self._frame_offsets[idx])

    if idx + 1 < len(self):
      frame_data = self._file.read(self._frame_offsets[idx + 1] - self._frame_offsets[idx])
    else:
      frame_data = self._file.read(self._filesize - self._frame_offsets[idx])

    frame_data = frame_data.decode("ascii").splitlines()

    frame = Frame(self.topology)
    frame.timestep = int(frame_data[1])
    frame.num_atoms = int(frame_data[3])
    
    xmin, xmax = map(float, frame_data[5].split())
    ymin, ymax = map(float, frame_data[6].split())
    zmin, zmax = map(float, frame_data[7].split())

    frame.box = BoxVolume(
        [xmin, ymin, zmin],
        [xmax, ymax, zmax]
    )

    columns = frame_data[8].split()[2:]

    column_map = {name: i for i, name in enumerate(columns)}

    if {"x", "y", "z"} <= column_map.keys():
      frame.positions = np.empty((frame.num_atoms, 3), dtype=np.float64)

    if {"vx", "vy", "vz"} <= column_map.keys():
      frame.velocities = np.empty((frame.num_atoms, 3), dtype=np.float64)

    if {"fx", "fy", "fz"} <= column_map.keys():
      frame.forces = np.empty((frame.num_atoms, 3), dtype=np.float64)
    
    if "id" in column_map:
      frame.ids = np.empty(frame.num_atoms, dtype=np.int32)

    if "mol" in column_map:
      frame.mol_ids = np.empty(frame.num_atoms, dtype=np.int32)

    if "type" in column_map:
      frame.types = np.empty(frame.num_atoms, dtype=np.int32)
    
    for i, line in enumerate(frame_data[9:]):
      fields = line.split()

      if frame.positions is not None:
        frame.positions[i, 0] = float(fields[column_map["x"]])
        frame.positions[i, 1] = float(fields[column_map["y"]])
        frame.positions[i, 2] = float(fields[column_map["z"]])

      if frame.velocities is not None:
        frame.velocities[i, 0] = float(fields[column_map["vx"]])
        frame.velocities[i, 1] = float(fields[column_map["vy"]])
        frame.velocities[i, 2] = float(fields[column_map["vz"]])

      if frame.forces is not None:
        frame.forces[i, 0] = float(fields[column_map["fx"]])
        frame.forces[i, 1] = float(fields[column_map["fy"]])
        frame.forces[i, 2] = float(fields[column_map["fz"]])

      if frame.ids is not None:
        frame.ids[i] = int(fields[column_map["id"]])

      if frame.mol_ids is not None:
        frame.mol_ids[i] = int(fields[column_map["mol"]])

      if frame.types is not None:
        frame.types[i] = int(fields[column_map["type"]])

    if frame.ids is not None:
      order = np.argsort(frame.ids)

      frame.ids = frame.ids[order]

      if frame.types is not None:
        frame.types = frame.types[order]

      if frame.mol_ids is not None:
        frame.mol_ids = frame.mol_ids[order]

      if frame.positions is not None:
        frame.positions = frame.positions[order]

      if frame.unwrapped_positions is not None:
        frame.unwrapped_positions = frame.unwrapped_positions[order]

      if frame.velocities is not None:
        frame.velocities = frame.velocities[order]

      if frame.forces is not None:
        frame.forces = frame.forces[order]

    return frame

  def __iter__(self):
    for idx in range(len(self)):
      yield self.read_frame(idx)

  def __len__(self):
    return len(self._frame_offsets)

  def _initialize(self):
    self._frame_offsets = []

    self._file.seek(0)

    while True:
      position = self._file.tell()
      line = self._file.readline()

      if not line:
        break

      if line.startswith(b"ITEM: TIMESTEP"):
        self._frame_offsets.append(position)

    self._file.seek(0)

  def _printline_at_byte(self, byte_position):
    self._file.seek(byte_position)
    print(self._file.readline().decode("ascii"), end="")

  def close(self):
    if not self._file.closed:
      self._file.close()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def __del__(self):
    self.close()