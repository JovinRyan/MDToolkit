import scipy.constants as sc
import pandas as pd
from MDToolkit.utils.misc_utils import is_real_float, is_strict_int
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

def read_elements_csv(file_path = "/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/PubChemElements_all.csv"):
    '''
    INPUT: \n
    '''
    elements_df = pd.read_csv(file_path)

    return elements_df

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

    base_header_values = ["atom_type", "atom_index", "atom_species", "molecule_name", "chain_id", "molecule_index", "x", "y", "z"]
    sample_df_values_list = sample_df_line.tolist()
    num_columns = len(sample_df_values_list)

    elemental_df = read_elements_csv()

    if num_columns < 9:
        error_message = f"Expected at least 9 columns in the PDB dataframe, but got {num_columns}. Please check the PDB file format."
        raise ValueError(error_message)
    elif num_columns > 9:
        for i in range(9, num_columns):
            base_header_values.append(f"extra_column_{i-8}")

    #Initializing variables
    header = [""] * num_columns
    atom_type_index = None
    atom_index_index = None
    atom_species_index = None
    molecule_name_index = None
    chain_id_index = None
    molecule_index_index = None
    x_index = None
    y_index = None
    z_index = None

    for i in range(num_columns):
        match sample_df_values_list[i]:
            case "ATOM" | "HETATM":
                atom_type_index = i
                atom_index_index = i + 1 if i + 1 < num_columns else None
            case _ if i == atom_index_index:
                pass
            case _ if is_strict_int(sample_df_values_list[i]) and atom_index_index is None:
                atom_index_index = i
            case _ if is_strict_int(sample_df_values_list[i]) and atom_index_index is not None:
                molecule_index_index = i
            case _ if is_real_float(sample_df_values_list[i]) and x_index is None and y_index is None and z_index is None:
                x_index = i
                y_index = i + 1 if i + 1 < num_columns else None
                z_index = i + 2 if i + 2 < num_columns else None
            case _ if sample_df_values_list[i] in elemental_df["Symbol"].values and atom_species_index is None:
                atom_species_index = i
            case _ if molecule_name_index is None:
                molecule_name_index = i
            case _ if chain_id_index is None:
                chain_id_index = i


    header[atom_type_index] = "atom_type" if atom_type_index is not None else ""
    header[atom_index_index] = "atom_index" if atom_index_index is not None else ""
    header[atom_species_index] = "atom_species" if atom_species_index is not None else ""
    header[molecule_index_index] = "molecule_index" if molecule_index_index is not None else ""
    header[x_index] = "x"
    header[y_index] = "y"
    header[z_index] = "z"
    header[molecule_name_index] = "molecule_name" if molecule_name_index is not None else ""
    header[chain_id_index] = "chain_id" if chain_id_index is not None else ""

    return header
