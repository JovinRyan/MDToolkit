import pandas as pd
from MDToolkit.IO.read_file import cif_file_to_structured_system, pdb_file_to_structured_system
from MDToolkit.data.objects import *
from MDToolkit.utils.structure_file_utils import create_periodic_images, delete_atoms_outside_region, create_elements_dictionary
from MDToolkit.IO.write_file import write_pdb_file_from_StructuredSystem, write_lammps_structure_file_atomic_full
from MDToolkit.custom_systems.membranes import cif_file_to_monolayer_membrane
from MDToolkit.utils.misc_utils import file_path_to_elements_and_stoichiometries
from MDToolkit.utils.cutom_systems_utils import find_center_of_mass

file_path_MoS2 = '/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/cif_files/MoS2.cif'

file_path_H20 = '/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/common_pdb_files/H2O.pdb'

file_path_MoS2_monolayer = '/home/jovinryanj/projects/mdtoolkit/Output/MoS2_monolayer.pdb'

# monolayer_system = cif_file_to_monolayer_membrane(file_path_MoS2)

# write_pdb_file_from_StructuredSystem(monolayer_system, file_name="MoS2_monolayer.pdb")

MoS2_monolayer_system = pdb_file_to_structured_system(file_path_MoS2_monolayer)

# print(MoS2_monolayer_system)

# com = find_center_of_mass(MoS2_monolayer_system)

# print(com)

write_lammps_structure_file_atomic_full(MoS2_monolayer_system, file_name="MoS2_monolayer.data")
