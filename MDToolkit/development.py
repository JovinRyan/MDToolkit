import pandas as pd
from MDToolkit.custom_systems.liquids import create_water_box
from MDToolkit.IO.write_file import write_lammps_structure_file_atomic_full

file_path_MoS2 = '/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/cif_files/MoS2.cif'

file_path_H2O = '/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/common_pdb_files/H2O.pdb'

file_path_MoS2_monolayer = '/home/jovinryanj/projects/mdtoolkit/Output/MoS2_monolayer.pdb'

water_box_dimensions = {
  "min_x" : 0.0, "max_x" : 100.0,
  "min_y" : 0.0, "max_y" : 100.0,
  "min_z" : 0.0, "max_z" : 100.0,
}

system = create_water_box(water_box_dimensions)

write_lammps_structure_file_atomic_full(system, file_name="H2O_vdw_centered.data")
