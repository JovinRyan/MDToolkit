import pandas as pd
from itertools import groupby
from MDToolkit.utils.structure_file_utils import *
from MDToolkit.utils.misc_utils import check_unique, is_real_float, is_strict_int
from MDToolkit.data.objects import *


def read_pdb(file_path : str):
    """
    Reads a PDB file and returns a pandas DataFrame containing the ATOM and HETATM lines.

    INPUT:\n
    file_path (str): The path to the PDB file.

    RETURNS:\n
    pdb_molecule_list (list): A list of molecule objects constructed from the PDB file.
    """

    start_index, end_index = identify_pdb_atom_indexes(file_path)

    pdb_df = pd.read_csv(file_path, skiprows=int(start_index), nrows=int(end_index - start_index), header=None, sep=r'\s+', engine="python-fwf")
    sample_line = pdb_df.iloc[0]

    pdb_df.columns = give_pdb_df_header(sample_line)

    # header rectification (does not work for all cases, but is a quick fix for some common PDB formatting issues)

    if not check_unique(pdb_df["chain_id"]) and check_unique(pdb_df["molecule_name"]):
        pdb_df.rename(columns={"chain_id": "molecule_name", "molecule_name": "chain_id"}, inplace=True)

    pdb_molecule_list = construct_molecule_list_from_df(pdb_df)

    return pdb_molecule_list

def read_packmol_pdb(file_path : str):
    """
    """

    start_index, end_index = identify_pdb_atom_indexes(file_path)
    col_widths = [6, 7, 2, 5, 2, 8, 8, 8, 8]
    pdb_df = pd.read_fwf(file_path, skiprows=int(start_index), nrows=int(end_index - start_index), header = None, widths=col_widths)

    pdb_df.columns = give_pdb_df_header(pdb_df.iloc[0])

    if not check_unique(pdb_df["chain_id"]) and check_unique(pdb_df["molecule_name"]):
        pdb_df.rename(columns={"chain_id": "molecule_name", "molecule_name": "chain_id"}, inplace=True)

    pdb_molecule_list = construct_molecule_list_from_df(pdb_df)

    return pdb_molecule_list


def read_cif(file_path):
    """
    Reads a CIF file and returns a pandas DataFrame containing the atomic information.

    INPUT:\n
    file_path (str): The path to the CIF file.

    RETURNS:\n
    cif_df (pandas DataFrame): A DataFrame containing the atomic information from the CIF file, with appropriate column headers.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()

        loop_indices = [i for i, line in enumerate(lines) if line.strip() == "loop_"]
        data_name_indices = [i for i, line in enumerate(lines) if line.strip().startswith("_")]

        for i in range(len(lines)):
            match lines[i].strip().split()[0]:
                case "_cell_length_a" : max_x = float(lines[i].strip().split()[1])
                case "_cell_length_b" : max_y = float(lines[i].strip().split()[1])
                case "_cell_length_c" : max_z = float(lines[i].strip().split()[1])
                case "_cell_angle_alpha" : cell_angle_alpha = float(lines[i].strip().split()[1])
                case "_cell_angle_beta" : cell_angle_beta = float(lines[i].strip().split()[1])
                case "_cell_angle_gamma" : cell_angle_gamma = float(lines[i].strip().split()[1])
                case "_chemical_formula_structural" : molecule_name = lines[i].strip().split()[1]

        # print(lines[loop_indices[-1] + 1:data_name_indices[-1] + 1])

        sanitized_lines = [line.strip() for line in lines[loop_indices[-1] + 1:data_name_indices[-1] + 1]]

    cif_df = pd.read_csv(file_path, skiprows=int(data_name_indices[-1]+1), sep = '\s+', names = sanitized_lines)

    cif_df["molecule_name"] = molecule_name
    cif_df["atom_type"] = "ATOM"

    atom_site_label_list = cif_df["_atom_site_label"].tolist()

    element_species_list = []
    atom_index_list = []
    for label in atom_site_label_list:
        split_list = ["".join(group) for key, group in groupby(label, key=str.isdigit)]
        try:
            split_list = [int(x) if is_strict_int(x) else x for x in split_list]
        except:
            pass
        element_species_list.append(split_list[0])
        atom_index_list.append(split_list[1])

    # convert zero indexing to 1-indexing
    if min(atom_index_list) == 0:
        atom_index_list = [x + 1 for x in atom_index_list]

    cif_df["atom_species"] = element_species_list
    cif_df["atom_index"] = atom_index_list
    cif_df["chain_id"] = "A"
    cif_df["molecule_index"] = 1
    cif_df["x"] = cif_df["_atom_site_fract_x"] * max_x
    cif_df["y"] = cif_df["_atom_site_fract_y"] * max_y
    cif_df["z"] = cif_df["_atom_site_fract_z"] * max_z

    cif_df.drop(columns=[col for col in cif_df.columns if col.startswith("_")], inplace=True)

    cif_molecule_list = construct_molecule_list_from_df(cif_df)

    return cif_molecule_list

def read_cif_box_dimensions_and_angles(file_path):
    """
    Reads a CIF file and extracts the box dimensions and angles.
    INPUT:\n
    file_path (str): The path to the CIF file.
    RETURNS:\n
    tuple: A tuple containing the box dimensions and angles: (box_dimensions, box_angles)
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

        loop_indices = [i for i, line in enumerate(lines) if line.strip() == "loop_"]
        data_name_indices = [i for i, line in enumerate(lines) if line.strip().startswith("_")]

        for i in range(len(lines)):
            match lines[i].strip().split()[0]:
                case "_cell_length_a" : max_x = float(lines[i].strip().split()[1])
                case "_cell_length_b" : max_y = float(lines[i].strip().split()[1])
                case "_cell_length_c" : max_z = float(lines[i].strip().split()[1])
                case "_cell_angle_alpha" : cell_angle_alpha = float(lines[i].strip().split()[1])
                case "_cell_angle_beta" : cell_angle_beta = float(lines[i].strip().split()[1])
                case "_cell_angle_gamma" : cell_angle_gamma = float(lines[i].strip().split()[1])

    box_dimensions = {
        "min_x": 0.0,
        "max_x": max_x,
        "min_y": 0.0,
        "max_y": max_y,
        "min_z": 0.0,
        "max_z": max_z
    }
    box_angles = {
        "angle_ab": cell_angle_alpha,
        "angle_ac": cell_angle_beta,
        "angle_bc": cell_angle_gamma
    }
    return box_dimensions, box_angles

def cif_file_to_structured_system(file_path):
    """
    Reads a CIF file and returns a StructuredSystem object containing the molecular system information.

    INPUT:\n
    file_path (str): The path to the CIF file.

    RETURNS:\n
    structured_system (StructuredSystem): A StructuredSystem object containing the molecular system information from the CIF file.
    """

    cif_molecule_list = read_cif(file_path)

    bounding_box, bounding_box_angles = read_cif_box_dimensions_and_angles(file_path)

    structured_system = StructuredSystem(molecule_list=cif_molecule_list, box_dimensions=bounding_box, box_angles=bounding_box_angles)

    return structured_system

def pdb_file_to_structured_system(file_path):
    """
    Reads a PDB file and returns a StructuredSystem object containing the molecular system information.

    INPUT:\n
    file_path (str): The path to the PDB file.

    RETURNS:\n
    structured_system (StructuredSystem): A StructuredSystem object containing the molecular system information from the PDB file.
    """

    pdb_molecule_list = read_pdb(file_path)

    box_dimensions = get_bounding_box_from_molecule_list(pdb_molecule_list)

    box_angles = get_bounding_box_angles_from_bounding_box(box_dimensions)

    structured_system = StructuredSystem(molecule_list=pdb_molecule_list, box_dimensions=box_dimensions, box_angles=box_angles)

    return structured_system

def packmol_pdb_file_to_structured_system(file_path):
    """
    Reads a Packmol PDB file and returns a StructuredSystem object containing the molecular system information.

    INPUT:\n
    file_path (str): The path to the Packmol PDB file.

    RETURNS:\n
    structured_system (StructuredSystem): A StructuredSystem object containing the molecular system information from the PDB file.
    """

    pdb_molecule_list = read_packmol_pdb(file_path)

    box_dimensions = get_bounding_box_from_molecule_list(pdb_molecule_list)

    box_angles = get_bounding_box_angles_from_bounding_box(box_dimensions)

    structured_system = StructuredSystem(molecule_list=pdb_molecule_list, box_dimensions=box_dimensions, box_angles=box_angles)

    structured_system.reset_molecule_ids()
    structured_system.reset_atom_ids()

    return structured_system
