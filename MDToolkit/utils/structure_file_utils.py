import scipy.constants as sc
# import mdtoolkit.logging as log

def estimate_number_density(density: float, molecular_weight : float, atom_count : int = 1):
    '''
    INPUT: \n
    density (float) : density of species in g/cm^3 \n
    molecular_weight (float) : molecular weight of species in g/mol \n
    atom_count (int) : number of atoms in species molecule. Default = 1 \n

    RETURNS: \n
    number_density (float) : estimated number density of species in atoms/angstom^3
    '''

    number_density = density/molecular_weight * sc.N_A/10**24 * atom_count

    return number_density

def identify_pdb_atom_indexes(file_path):
    '''
    Reads a PDB file and identifies the indexes of the ATOM and HETATM lines.

    INPUT:
    file_path (str): The path to the PDB file.

    RETURNS: \n
    tuple: A tuple containing start and end indexes: (start_index, end_index)
    '''

    with open(file_path, 'r') as f:
        lines = f.readlines()
        start_index_ATOM = next((i for i, line in enumerate(lines) if line.startswith("ATOM")), float('inf'))
        start_index_HETATM = next((i for i, line in enumerate(lines) if line.startswith("HETATM")), float('inf'))
        end_index = next((i for i, line in enumerate(lines) if line.startswith("END")), float('inf'))
    return (min(start_index_ATOM, start_index_HETATM), end_index)

def give_pdb_df_header(sample_df_line):
    '''
    INPUT: \n
    sample_df_line (pandas Series): A sample line from the PDB dataframe to determine the number of columns.
    RETURNS: \n
    header (list): A list of column names for the PDB dataframe, based on the number of columns in the sample line.
    '''

    base_header = ["atom_type", "atom_index", "atom_species", "molecule_name", "chain_id", "molecule_index", "x", "y", "z"]
    num_columns = len(sample_df_line.tolist())

    if num_columns < 9:
        error_message = f"Expected at least 9 columns in the PDB dataframe, but got {num_columns}. Please check the PDB file format."
        raise ValueError(error_message)
    elif num_columns > 9:
        for i in range(9, num_columns):
            base_header.append(f"extra_column_{i-8}")

    return base_header
