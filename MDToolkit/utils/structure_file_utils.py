import scipy.constants as sc
# import mdtoolkit.logging as log

def estimate_number_density(density: float, molecular_weight : float, atom_count : int = 1):
    '''
    INPUT: \n
    density (float) : density of species in g/cm^3 \n
    molecular_weight (float) : molecular weight of species in g/mol \n
    atom_count (int) : number of atoms in species molecule. Default = 1 \n
    '''

    number_density = density/molecular_weight * sc.N_A/10**24 * atom_count

    return number_density

def identify_pdb_atom_indexes(file_path):
    '''
    Reads a PDB file and identifies the indexes of the ATOM and HETATM lines.

    Parameters:
    file_path (str): The path to the PDB file.

    Returns:
    tuple: A tuple containing start and end indexes: (start_index, end_index)
    '''

    with open(file_path, 'r') as f:
        lines = f.readlines()
        start_index_ATOM = lines.find("ATOM")
        start_index_HETATM = lines.find("HETATM")
        end_index = lines.find("END")
    return (min(start_index_ATOM, start_index_HETATM), end_index)
