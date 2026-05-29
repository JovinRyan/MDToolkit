import pandas as pd
from MDToolkit.custom_systems.liquids import create_water_box, create_ionic_liquid_box
from MDToolkit.IO.write_file import write_lammps_structure_file_atomic_full
from MDToolkit.IO.read_file import pdb_file_to_structured_system

file_path_OMIM = "/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/my_pdb_files/OMIM.pdb"
file_path_PF6 = "/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/my_pdb_files/PF6.pdb"

box_dimensions = {
  "min_x" : 0.0, "max_x" : 50,
  "min_y" : 0.0, "max_y" : 50,
  "min_z" : 0.0, "max_z" : 50,
}

OMIM_PF6_system = create_ionic_liquid_box(box_dimensions, file_path_OMIM, file_path_PF6, ionic_liquid_density = 1.24)

write_lammps_structure_file_atomic_full(OMIM_PF6_system, file_name = "OMIMPF6_50by50by50.data")
