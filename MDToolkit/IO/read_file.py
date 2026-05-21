import pandas as pd
from MDToolkit.utils.structure_file_utils import identify_pdb_atom_indexes, give_pdb_df_header
from MDToolkit.utils.misc_utils import check_unique

def read_pdb(file_path):
    """
    Reads a PDB file and returns a pandas DataFrame containing the ATOM and HETATM lines.

    INPUT:\n
    file_path (str): The path to the PDB file.

    RETURNS:\n
    pdb_df (pandas DataFrame): A DataFrame containing the ATOM and HETATM lines from the PDB file, with appropriate column headers.
    """

    start_index, end_index = identify_pdb_atom_indexes(file_path)

    pdb_df = pd.read_csv(file_path, skiprows=int(start_index), nrows=int(end_index - start_index), header=None, sep='\s+')
    sample_line = pdb_df.iloc[0]

    pdb_df.columns = give_pdb_df_header(sample_line)

    # header rectification (does not work for all cases, but is a quick fix for some common PDB formatting issues)

    if not check_unique(pdb_df["chain_id"]) and check_unique(pdb_df["molecule_name"]):
        pdb_df.rename(columns={"chain_id": "molecule_name", "molecule_name": "chain_id"}, inplace=True)

    return pdb_df
