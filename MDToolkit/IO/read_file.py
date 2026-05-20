import pandas as pd
from utils.structure_file_utils import identify_pdb_atom_indexes

def read_pdb(file_path):
    """
    Reads a PDB file and returns its contents as a string.

    Parameters:
    file_path (str): The path to the PDB file.

    Returns:
    str: The contents of the PDB file.
    """

    start_index, end_index = identify_pdb_atom_indexes(file_path)

    pdb_df = pd.read_csv(file_path, sep='\t', skiprows=start_index + 1, nrows=end_index - start_index - 1, header=None)

    return pdb_df

print(read_pdb('/home/jovinryanj/projects/MDToolkit/data/common_pdb_files/H2O.pdb'))
