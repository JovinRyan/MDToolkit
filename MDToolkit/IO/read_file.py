import pandas as pd
from itertools import groupby
from io import StringIO
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from MDToolkit.utils.structure_file_utils import *
from MDToolkit.utils.misc_utils import check_unique, is_real_float, is_strict_int
from MDToolkit.data.objects import Simulation, StructuredSystem, Atom, Molecule, construct_molecule_list_from_df


def read_pdb(file_path : str):
    """
    Reads a PDB file and returns a pandas DataFrame containing the ATOM and HETATM lines.

    INPUT:\n
    file_path (str): The path to the PDB file.

    RETURNS:\n
    pdb_molecule_list (list): A list of molecule objects constructed from the PDB file.
    """

    start_index, end_index = identify_pdb_atom_indexes(file_path)

    colspecs = [
        (0, 6),    # record_type
        (6, 11),   # atom_index
        (12, 16),  # atom_species
        (17, 20),  # molecule_name
        (21, 22),  # chain_id
        (22, 26),  # molecule_index
        (30, 38),  # x
        (38, 46),  # y
        (46, 54),  # z
        (76, 78),  # element
    ]

    column_names = [
        "atom_type",
        "atom_index",
        "atom_species_verbose",
        "molecule_name",
        "chain_id",
        "molecule_index",
        "x",
        "y",
        "z",
        "atom_species"
    ]

    pdb_df = pd.read_fwf(
        file_path,
        colspecs=colspecs,
        names=column_names,
        skiprows=int(start_index),
        nrows=int(end_index - start_index)
    )

    pdb_df["atom_type"] = pdb_df["atom_type"].str.strip()

    pdb_df["atom_species"] = pdb_df["atom_species"].str.strip()

    pdb_df["molecule_name"] = pdb_df["molecule_name"].str.strip()

    pdb_df["chain_id"] = pdb_df["chain_id"].astype(str).str.strip()

    pdb_df["atom_index"] = pdb_df["atom_index"].astype(int)

    pdb_df["molecule_index"] = pdb_df["molecule_index"].astype(int)

    pdb_df["x"] = pdb_df["x"].astype(float)

    pdb_df["y"] = pdb_df["y"].astype(float)

    pdb_df["z"] = pdb_df["z"].astype(float)

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

def lammps_data_file_to_structured_system(file_path):
    '''
    '''
    indices_dict = get_lammps_data_file_indexes(file_path)
    atom_indices = indices_dict["atoms"]
    bond_indices = indices_dict["bonds"]
    angle_indices = indices_dict["angles"]
    mass_indices = indices_dict["masses"]
    box_dims_indices = indices_dict["box_dims"]

    elemental_data_df = read_elements_csv()

    with open(file_path, "r") as f:
        lines = f.readlines()
        mass_lines = lines[mass_indices[0]:mass_indices[1] + 1]
        box_dims_lines = lines[box_dims_indices[0]:box_dims_indices[1] + 1]

    mass_inttype_mapping = {}
    for i in range(len(mass_lines)):
        entry = {int(mass_lines[i].strip().split(" ")[0]) : float(mass_lines[i].strip().split(" ")[1])}
        mass_inttype_mapping.update(entry)
    
    box_dims_dict = {}
    for i in range(len(box_dims_lines)):
        entry_min = {"min_" + box_dims_lines[i].strip().split()[2][0] : float(box_dims_lines[i].strip().split()[0])}
        entry_max = {"max_" + box_dims_lines[i].strip().split()[2][0] : float(box_dims_lines[i].strip().split()[1])}
        box_dims_dict.update(entry_min)
        box_dims_dict.update(entry_max)
        

    symbol_inttype_mapping = {}
    for atom_type, mass in mass_inttype_mapping.items():
        closest_idx = (elemental_data_df["AtomicMass"] - mass).abs().idxmin()
        symbol = elemental_data_df.loc[closest_idx, "Symbol"]
        symbol_inttype_mapping[atom_type] = symbol

    atom_cols = ["atom_id", "mol_id", "type", "charge", "x", "y", "z", "flag1", "flag2", "flag3"]
    atoms_df = pd.read_csv(file_path, sep = " ", skiprows = atom_indices[0], nrows = atom_indices[1]-atom_indices[0] + 1, names = atom_cols)
    atoms_df = atoms_df.sort_values(by = ["mol_id", "atom_id"]).reset_index(drop = True)
    atoms_df.drop(["flag1", "flag2", "flag3"], axis = 1)

    molecule_list = []

    for mol_id, molecule_df in atoms_df.groupby("mol_id"):
        molecule_df.reset_index(inplace = True, drop = True)
        atoms_list = []
        for i in range(len(molecule_df)):
            atoms_list.append(Atom(id = int(molecule_df.iloc[i]["atom_id"]), element = symbol_inttype_mapping[molecule_df.iloc[i]["type"]], position=[molecule_df.iloc[i]["x"], molecule_df.iloc[i]["y"], molecule_df.iloc[i]["z"]], charge=molecule_df.iloc[i]["charge"]))         
        
        molecule_list.append(Molecule(molecule_id=mol_id,molecule_name= "ABC", atoms=atoms_list))

    return StructuredSystem(molecule_list = molecule_list, box_dimensions = box_dims_dict)

def _process_frame(df, type_mapping, coordinate_type = "standard"):
    '''
    Helper function to process a single frame into molecules.
    '''
    molecules_list = []
    tm = type_mapping

    for mol_id, mol_df in df.groupby("mol", sort=False):
        ids = mol_df["id"].to_numpy()
        types = mol_df["type"].to_numpy()
        if coordinate_type == "unwrapped":
            xs = mol_df["xu"].to_numpy()
            ys = mol_df["yu"].to_numpy()
            zs = mol_df["zu"].to_numpy()
            qs = mol_df["q"].to_numpy()
        else:
            xs = mol_df["x"].to_numpy()
            ys = mol_df["y"].to_numpy()
            zs = mol_df["z"].to_numpy()
            qs = mol_df["q"].to_numpy()

        atoms_list = [Atom(int(i), tm[t], [x, y, z], q)for i, t, x, y, z, q in zip(ids, types, xs, ys, zs, qs)]

        molecules_list.append(Molecule(molecule_id=mol_id, molecule_name="ABC", atoms=atoms_list))

    return molecules_list


def lammps_dump_file_to_simulation(file_path, type_mapping: dict, coordinate_type = "standard", n_jobs=-1):
    '''
    Parallelized LAMMPS dump file parser → Simulation object.
    '''

    idxs_dict = get_lammps_dump_file_indices(file_path)
    timesteps_idxs = idxs_dict["timesteps"]
    atoms_idxs = idxs_dict["atoms"]
    atom_counts_idxs = idxs_dict["atom_counts"]
    box_bounds_idxs = idxs_dict["box_bounds"]

    with open(file_path, "r") as f:
        lines = f.readlines()

    atom_counts_list = [int(lines[idx].strip()) for idx in atom_counts_idxs]
    timesteps_list = [int(lines[idx].strip()) for idx in timesteps_idxs]

    box_bounds_list = []
    for idx in box_bounds_idxs:
        min_x, max_x = lines[idx].split()
        min_y, max_y = lines[idx + 1].split()
        min_z, max_z = lines[idx + 2].split()

        box_bounds_list.append({
            "min_x": float(min_x), "max_x": float(max_x),
            "min_y": float(min_y), "max_y": float(max_y),
            "min_z": float(min_z), "max_z": float(max_z)
        })

    header_line = lines[atoms_idxs[0] - 1]
    columns = header_line.removeprefix("ITEM: ATOMS ").split()

    atom_dfs = []
    for i in range(len(atoms_idxs)):
        start = atoms_idxs[i]
        end = start + atom_counts_list[i]

        atom_lines = lines[start:end]

        df = pd.read_csv(
            StringIO("".join(atom_lines)),
            names=columns,
            sep=r"\s+",
            engine="c"
        )

        df.sort_values("id", inplace=True)
        atom_dfs.append(df)

    # -----------------------------
    # PARALLEL FRAME PROCESSING
    # -----------------------------
    molecule_lists_list = Parallel(n_jobs=n_jobs)(
        delayed(_process_frame)(df, type_mapping, coordinate_type)
        for df in tqdm(atom_dfs, desc="Processing frames", unit="frame")
    )

    structured_systems_list = [
        StructuredSystem(molecule_lists_list[i], box_dimensions=box_bounds_list[i])
        for i in range(len(molecule_lists_list))
    ]

    return Simulation(structured_systems_list, timesteps_list, atom_counts_list)