import pandas as pd
from MDToolkit.IO.read_file import read_pdb
from MDToolkit.data.objects import construct_molecule_list_from_df, StructuredSystem
from MDToolkit.utils.structure_file_utils import get_bounding_box_from_molecule_list, get_bounding_box_angles_from_bounding_box
from MDToolkit.IO.write_file import write_pdb_file_from_StructuredSystem

file_path = '/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/my_pdb_files/BMIM.pdb'
file_path2 = '/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/common_pdb_files/H2O.pdb'

pdb_df = read_pdb(file_path)
molecule_list = construct_molecule_list_from_df(pdb_df)

molecule_list[0].rotate_molecule_spherical(0, 90, 0)

# print(molecule_list[0])

system = StructuredSystem(molecule_list, box_dimensions = get_bounding_box_from_molecule_list(molecule_list), box_angles = get_bounding_box_angles_from_bounding_box(get_bounding_box_from_molecule_list(molecule_list)))

# write_pdb_file_from_StructuredSystem(system, file_name = "H2O_rotated.pdb", file_path="./Output")
write_pdb_file_from_StructuredSystem(system, file_name = "BMIM_rotated.pdb", file_path="./Output")

