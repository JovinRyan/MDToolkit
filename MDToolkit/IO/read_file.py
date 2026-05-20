import pandas as pd
from MDToolkit.utils.structure_file_utils import identify_pdb_atom_indexes, give_pdb_df_header

def read_pdb(file_path):
    """
    Reads a PDB file and returns its contents as a string.

    Parameters:
    file_path (str): The path to the PDB file.

    Returns:
    str: The contents of the PDB file.
    """

    start_index, end_index = identify_pdb_atom_indexes(file_path)

    pdb_df = pd.read_csv(file_path, skiprows=int(start_index), nrows=int(end_index - start_index), header=None, sep='\s+')
    sample_line = pdb_df.iloc[0]

    pdb_df.columns = give_pdb_df_header(sample_line)

    return pdb_df

print(read_pdb('/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/common_pdb_files/H2O.pdb'))
