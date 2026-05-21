import pandas as pd
from MDToolkit.IO.read_file import read_pdb
from MDToolkit.data.objects import construct_molecule_list_from_df

file_path = '/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/my_pdb_files/BMIM.pdb'
file_path2 = '/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/common_pdb_files/H2O.pdb'

pdb_df = read_pdb(file_path)
molecule_list = construct_molecule_list_from_df(pdb_df)

print(molecule_list)


