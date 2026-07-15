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
    self.stoich_dict = {key : 1.0 for key in type_mapping.keys()}
  
  def rewrite_charges(self, charges_dict : dict):
    for key in charges_dict.keys():
      if key in self.charges.keys():
        self.charges[key] = charges_dict[key]
  
  def get_bond_types(self):
    if self.bonds is None:
      return None 
    else:
      types = np.unique(self.bonds[:, 1])
      return len(types)
  
  def get_angle_types(self):
    if self.angles is None:
      return None 
    else:
      types = np.unique(self.angles[:, 1])
      return len(types)


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
  
  def populate_stoich_dict(self):
    for key in self.topology.stoich_dict:
      self.topology.stoich_dict[key] = (self.types == key).sum()
    
    min_count = min(self.topology.stoich_dict.values())

    for key in self.topology.stoich_dict:
      self.topology.stoich_dict[key] = self.topology.stoich_dict[key] / min_count


  def set_box_from_positions(self, buffer = [0, 0, 0]):
    min_x = min(self.positions[:, 0])
    max_x = max(self.positions[:, 0])

    min_y = min(self.positions[:, 1])
    max_y = max(self.positions[:, 1])

    min_z = min(self.positions[:, 2])
    max_z = max(self.positions[:, 2])

    self.box = BoxVolume([min_x - buffer[0] / 2, min_y - buffer[1] / 2, min_z - buffer[2] / 2], [max_x + buffer[0] / 2, max_y + buffer[1] / 2, max_z + buffer[2] / 2])

  def replicate(self, images = [3, 3, 3]):
    '''
    '''

    images = np.asarray(images, dtype = np.int32)

    if np.any(images < 1):
        raise ValueError(
            "All image counts must be at least 1."
        )

    translations = [
        self.box.image_translation([i, j, k])
        for i in range(images[0])
        for j in range(images[1])
        for k in range(images[2])
    ]

    supercell = Frame(self.topology)

    supercell.timestep = self.timestep
    supercell.box = self.box.replicate(images)

    n_images = len(translations)

    supercell.num_atoms = self.num_atoms * n_images

    supercell.positions = np.concatenate([
        self.positions + translation
        for translation in translations
    ])

    supercell.ids = np.arange(
        1,
        supercell.num_atoms + 1,
        dtype = np.int32
    )

    if self.types is not None:

        supercell.types = np.tile(
            self.types,
            n_images
        )

    if self.mol_ids is not None:

        supercell.mol_ids = np.tile(
            self.mol_ids,
            n_images
        )

    if self.unwrapped_positions is not None:

        supercell.unwrapped_positions = np.concatenate([
            self.unwrapped_positions + translation
            for translation in translations
        ])

    if self.velocities is not None:

        supercell.velocities = np.tile(
            self.velocities,
            (n_images, 1)
        )

    if self.forces is not None:

        supercell.forces = np.tile(
            self.forces,
            (n_images, 1)
        )

    return supercell
  
  def set_center(self, center = [0, 0, 0]):
    '''
    '''

    center = np.asarray(center, dtype=float)

    current_center = (
        self.box.origin
        + 0.5 * np.sum(self.box.lattice_vectors, axis=1)
    )

    translation = center - current_center

    self.positions += translation

    if self.unwrapped_positions is not None:
        self.unwrapped_positions += translation

    self.box.set_center(center)
  
  def delete_atoms_outside_ranges(self, x_range = None, y_range = None, z_range = None):
    '''
    '''
    mask = np.ones(self.num_atoms, dtype=bool)

    if x_range is not None:
      mask &= (self.positions[:, 0] >= x_range[0]) & (self.positions[:, 0] <= x_range[1])

    if y_range is not None:
      mask &= (self.positions[:, 1] >= y_range[0]) & (self.positions[:, 1] <= y_range[1])

    if z_range is not None:
      mask &= (self.positions[:, 2] >= z_range[0]) & (self.positions[:, 2] <= z_range[1])

    if self.ids is not None:
      self.ids = self.ids[mask]

    if self.types is not None:
      self.types = self.types[mask]

    if self.mol_ids is not None:
      self.mol_ids = self.mol_ids[mask]

    if self.positions is not None:
      self.positions = self.positions[mask]

    if self.unwrapped_positions is not None:
      self.unwrapped_positions = self.unwrapped_positions[mask]

    if self.velocities is not None:
      self.velocities = self.velocities[mask]

    if self.forces is not None:
      self.forces = self.forces[mask]

    self.num_atoms = np.sum(mask)

    if self.num_atoms > 0:
      self.box = BoxVolume(
          self.positions.min(axis=0),
          self.positions.max(axis=0))
  
  def rotate(self, angles, origin = None):
    '''
    '''

    angles = np.deg2rad(angles)

    x = angles[0]
    y = angles[1]
    z = angles[2]

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])

    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])

    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])

    rotation_matrix = Rz @ Ry @ Rx

    if origin is None:
      origin = (self.box.mins + self.box.maxs) / 2

    origin = np.asarray(origin)

    if self.positions is not None:
      self.positions = (
          self.positions - origin
      ) @ rotation_matrix.T + origin

    if self.unwrapped_positions is not None:
      self.unwrapped_positions = (
          self.unwrapped_positions - origin
      ) @ rotation_matrix.T + origin

    if self.velocities is not None:
      self.velocities = self.velocities @ rotation_matrix.T

    if self.forces is not None:
      self.forces = self.forces @ rotation_matrix.T

    self.box.rotate(
        rotation_matrix,
        origin
    )

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