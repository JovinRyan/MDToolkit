import pandas as pd
from MDToolkit.utils.structure_file_utils import identify_pdb_atom_indexes

file_path = '/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/common_pdb_files/H2O.pdb'

df = pd.read_csv(file_path)
print(df.iloc[4])
