import scipy.constants as sc
import pandas as pd
import numpy as np
import math
import copy
import json
from MDToolkit.utils.misc_utils import is_real_float, is_strict_int
from MDToolkit.paths import ELEMENTS_CSV, MOLECULAR_DATA_JSON


def estimate_number_density(density: float, molecular_weight : float):
    '''
    INPUT: \n
    density (float) : density of species in g/cm^3 \n
    molecular_weight (float) : molecular weight of species in g/mol \n

    RETURNS: \n
    number_density (float) : estimated number density of species in molecules/angstom^3
    '''

    number_density = density/molecular_weight * sc.N_A/10**24

    return number_density

def estimate_solute_solvent_number_density_from_molarity(
    molarity: float,
    solvent_molecular_weight: float,
    solute_molecular_weight: float,
    solution_density: float = 1.0,
):
    '''
    Estimate the number densities of the solvent and solute for a solution of
    known molarity.

    INPUT:
    molarity (float) : solute concentration in mol/L
    solvent_molecular_weight (float) : solvent molecular weight in g/mol
    solute_molecular_weight (float) : solute molecular weight in g/mol
    solution_density (float) : solution density in g/cm^3. Default = 1.0

    RETURNS:
    solvent_number_density (float) : solvent number density in molecules/angstrom^3
    solute_number_density (float) : solute number density in molecules/angstrom^3
    '''

    # Convert solution density from g/cm^3 to g/L
    solution_density *= 1000.0

    # Solvent concentration (mol/L)
    solvent_concentration = (
        solution_density - molarity * solute_molecular_weight
    ) / solvent_molecular_weight

    # Convert mol/L -> molecules/angstrom^3
    conversion = sc.Avogadro / 1e27

    solvent_number_density = solvent_concentration * conversion
    solute_number_density = molarity * conversion

    return solvent_number_density, solute_number_density

def molecular_formula_to_elements_and_stoichiometries(formula : str):
    char_list = list(formula)
    element_list = []
    stoich_list = []
    for i in range(len(char_list)):
        if char_list[i].isupper():
            element_list.append(char_list[i])
            stoich_list.append(1.0)
        elif char_list[i].islower():
            element_list[-1] += char_list[i]
        elif char_list[i].isdigit():
            try:
                if char_list[i + 1].isdigit():
                    char_list[i] += char_list[i+1]
                    char_list[i + 1] = ""
                stoich_list.append(float(char_list[i]))
                del stoich_list[-2]    
            except:
                stoich_list.append(float(char_list[i]))
                del stoich_list[-2]

    return element_list, stoich_list

def file_path_to_elements_and_stoichiometries(file_path : str):
    formula = file_path.split("/")[-1].split(".")[0]
    
    return molecular_formula_to_elements_and_stoichiometries(formula)

def elements_and_stoichiometries_to_molar_mass(elements: list[str], stoichiometries: list[float]):

    elements_dict = create_elements_dictionary()

    molar_mass = 0.0
    for i in range(len(elements)):
        molar_mass += elements_dict[elements[i]]["AtomicMass"] * stoichiometries[i]
    
    return molar_mass

def molecular_formula_to_molar_mass(formula: str):
    elements, stoichs = molecular_formula_to_elements_and_stoichiometries(formula)

    return elements_and_stoichiometries_to_molar_mass(elements, stoichs)

def read_elements_csv(file_path = ELEMENTS_CSV):
    '''
    INPUT: \n
    file_path (str) : The path to the CSV file containing elemental data. Default is set to a common location within the MDToolkit project.\n

    RETURNS: \n
    elements_df (pandas DataFrame) : A DataFrame containing elemental data from the specified CSV file.
    '''
    elements_df = pd.read_csv(file_path)

    return elements_df

def create_elements_dictionary(file_path = ELEMENTS_CSV):
    '''
    INPUT: \n
    file_path (str) : The path to the CSV file containing elemental data. Default is set to a common location within the MDToolkit project.\n

    RETURNS: \n
    elements_dict (dict) : A dictionary containing elemental data from the specified CSV file.\n

    DATA: \n
    "AtomicNumber","Symbol","Name","AtomicMass","CPKHexColor","ElectronConfiguration","Electronegativity","AtomicRadius","IonizationEnergy","ElectronAffinity","OxidationStates","StandardState","MeltingPoint","BoilingPoint","Density","GroupBlock","YearDiscovered"
    '''
    elements_df = read_elements_csv(file_path)
    elements_dict = elements_df.set_index('Symbol').T.to_dict()
    return elements_dict

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

def get_bounding_box_from_molecule_list(molecule_list):
    '''
    INPUT: \n
    molecule_list (list): A list of molecule objects, where each molecule object has a list of atom objects with x, y, z coordinates.

    RETURNS: \n
    bounding_box (dict): A dictionary containing the minimum and maximum x, y, z coordinates that define the bounding box of the molecule list.
    '''

    min_x = min(atom.position[0] for molecule in molecule_list for atom in molecule.atoms)
    max_x = max(atom.position[0] for molecule in molecule_list for atom in molecule.atoms)
    min_y = min(atom.position[1] for molecule in molecule_list for atom in molecule.atoms)
    max_y = max(atom.position[1] for molecule in molecule_list for atom in molecule.atoms)
    min_z = min(atom.position[2] for molecule in molecule_list for atom in molecule.atoms)
    max_z = max(atom.position[2] for molecule in molecule_list for atom in molecule.atoms)

    bounding_box = {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "min_z": min_z,
        "max_z": max_z
    }

    return bounding_box

def get_bounding_box_angles_from_bounding_box(bounding_box):
    '''
    INPUT: \n
    bounding_box (dict): A dictionary containing the minimum and maximum x, y, z coordinates that define the bounding box of a molecule or system.

    RETURNS: \n
    box_angles (dict): A dictionary containing the angles (in degrees) between the edges of the bounding box, assuming it is a parallelepiped.
    '''

    # Calculate edge vectors
    edge_vector_a = [bounding_box["max_x"] - bounding_box["min_x"], 0, 0]
    edge_vector_b = [0, bounding_box["max_y"] - bounding_box["min_y"], 0]
    edge_vector_c = [0, 0, bounding_box["max_z"] - bounding_box["min_z"]]

    # Calculate angles between edge vectors
    def calculate_angle(vec1, vec2):
        dot_product = sum(vec1[i] * vec2[i] for i in range(3))
        magnitude_vec1 = math.sqrt(sum(vec1[i] ** 2 for i in range(3)))
        magnitude_vec2 = math.sqrt(sum(vec2[i] ** 2 for i in range(3)))
        if magnitude_vec1 == 0 or magnitude_vec2 == 0:
            return 0.0
        cos_angle = dot_product / (magnitude_vec1 * magnitude_vec2)
        cos_angle = max(min(cos_angle, 1), -1)  # Clamp to avoid numerical issues
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    angle_ab = calculate_angle(edge_vector_a, edge_vector_b)
    angle_ac = calculate_angle(edge_vector_a, edge_vector_c)
    angle_bc = calculate_angle(edge_vector_b, edge_vector_c)

    box_angles = {
        "angle_ab": angle_ab,
        "angle_ac": angle_ac,
        "angle_bc": angle_bc
    }

    return box_angles

def identify_cif_atom_indexes(file_path):
    '''
    Reads a CIF file and identifies the indexes of the atomic information lines.

    INPUT:
    file_path (str): The path to the CIF file.

    RETURNS: \n
    tuple: A tuple containing start and end indexes: (start_index, end_index)
    '''

    with open(file_path, 'r') as f:
        lines = f.readlines()
        start_index = next((i for i, line in enumerate(lines) if line.startswith("loop_") and "_atom_site." in lines[i+1]), float('inf'))
        end_index = next((i for i, line in enumerate(lines) if line.startswith("loop_") and "_atom_site." in lines[i+1] and i > start_index), float('inf'))
    return (start_index + 2, end_index)

def create_periodic_images(StructuredSystem, image_vectors = (2, 2, 1)):
    '''
    Creates periodic images of a structured system based on the specified image vectors.

    INPUT: \n
    StructuredSystem (StructuredSystem): A structured system object containing molecule and atom information, as well as box dimensions and angles.
    image_vectors (tuple): A tuple specifying the number of periodic images to create in each dimension (x, y, z). Default is (2, 2, 1).

    RETURNS: \n
    new_structured_system (StructuredSystem): A new structured system object containing the original system and its periodic images.
    '''

    x_disp_vector = [i * StructuredSystem.box_dimensions["max_x"] - StructuredSystem.box_dimensions["min_x"] for i in range(image_vectors[0])]
    y_disp_vector = [i * StructuredSystem.box_dimensions["max_y"] - StructuredSystem.box_dimensions["min_y"] for i in range(image_vectors[1])]
    z_disp_vector = [i * StructuredSystem.box_dimensions["max_z"] - StructuredSystem.box_dimensions["min_z"] for i in range(image_vectors[2])]

    original_system = copy.deepcopy(StructuredSystem)
    Combined_system = copy.deepcopy(StructuredSystem)

    for x_disp in x_disp_vector:
        for y_disp in y_disp_vector:
            for z_disp in z_disp_vector:
                if x_disp == 0 and y_disp == 0 and z_disp == 0:
                    continue
                new_system = copy.deepcopy(original_system)
                for molecule in new_system.molecule_list:
                    for atom in molecule.atoms:
                        atom.position[0] += x_disp
                        atom.position[1] += y_disp
                        atom.position[2] += z_disp
                new_system.box_dimensions["min_x"] += x_disp
                new_system.box_dimensions["max_x"] += x_disp
                new_system.box_dimensions["min_y"] += y_disp
                new_system.box_dimensions["max_y"] += y_disp
                new_system.box_dimensions["min_z"] += z_disp
                new_system.box_dimensions["max_z"] += z_disp

                Combined_system.combine_with_other_structured_system(new_system)

    return Combined_system

def delete_molecules_in_region(StructuredSystem, region_bounds):
    '''
    Deletes molecules from a structured system that have any atoms within a specified region.

    INPUT: \n
    StructuredSystem (StructuredSystem): A structured system object containing molecule and atom information, as well as box dimensions and angles.
    region_bounds (dict): A dictionary specifying the bounds of the region in the format {"min_x": value, "max_x": value, "min_y": value, "max_y": value, "min_z": value, "max_z": value}.

    RETURNS: \n
    new_structured_system (StructuredSystem): A new structured system object with the specified molecules removed.
    '''

    new_structured_system = copy.deepcopy(StructuredSystem)
    new_molecule_list = []

    for molecule in new_structured_system.molecule_list:
        keep_molecule = True
        for atom in molecule.atoms:
            if (region_bounds["min_x"] <= atom.position[0] <= region_bounds["max_x"] and
                region_bounds["min_y"] <= atom.position[1] <= region_bounds["max_y"] and
                region_bounds["min_z"] <= atom.position[2] <= region_bounds["max_z"]):
                keep_molecule = False
                break
        if keep_molecule:
            new_molecule_list.append(molecule)

    new_structured_system.molecule_list = new_molecule_list

    return new_structured_system

def delete_atoms_in_region(StructuredSystem, region_bounds):
    '''
    Deletes atoms from a structured system that are within a specified region.

    INPUT: \n
    StructuredSystem (StructuredSystem): A structured system object containing molecule and atom information, as well as box dimensions and angles.
    region_bounds (dict): A dictionary specifying the bounds of the region in the format {"min_x": value, "max_x": value, "min_y": value, "max_y": value, "min_z": value, "max_z": value}.

    RETURNS: \n
    new_structured_system (StructuredSystem): A new structured system object with the specified atoms removed.
    '''

    new_structured_system = copy.deepcopy(StructuredSystem)

    for molecule in new_structured_system.molecule_list:
        new_atom_list = []
        for atom in molecule.atoms:
            if not (region_bounds["min_x"] <= atom.position[0] <= region_bounds["max_x"] and
                    region_bounds["min_y"] <= atom.position[1] <= region_bounds["max_y"] and
                    region_bounds["min_z"] <= atom.position[2] <= region_bounds["max_z"]):
                new_atom_list.append(atom)
        molecule.atoms = new_atom_list

    return new_structured_system

def delete_molecules_outside_region(StructuredSystem, region_bounds):
    '''
    Deletes molecules from a structured system that do not have any atoms within a specified region.

    INPUT: \n
    StructuredSystem (StructuredSystem): A structured system object containing molecule and atom information, as well as box dimensions and angles.
    region_bounds (dict): A dictionary specifying the bounds of the region in the format {"min_x": value, "max_x": value, "min_y": value, "max_y": value, "min_z": value, "max_z": value}.

    RETURNS: \n
    new_structured_system (StructuredSystem): A new structured system object with the specified molecules removed.
    '''

    new_structured_system = copy.deepcopy(StructuredSystem)
    new_molecule_list = []

    for molecule in new_structured_system.molecule_list:
        keep_molecule = False
        for atom in molecule.atoms:
            if (region_bounds["min_x"] <= atom.position[0] <= region_bounds["max_x"] and
                region_bounds["min_y"] <= atom.position[1] <= region_bounds["max_y"] and
                region_bounds["min_z"] <= atom.position[2] <= region_bounds["max_z"]):
                keep_molecule = True
                break
        if keep_molecule:
            new_molecule_list.append(molecule)

    new_structured_system.molecule_list = new_molecule_list

    return new_structured_system

def delete_atoms_outside_region(structured_system, region_bounds : dict, buffer = 2.5):
    '''
    Deletes atoms from a structured system that are outside a specified region.

    INPUT: \n
    StructuredSystem (StructuredSystem): A structured system object containing molecule and atom information, as well as box dimensions and angles.
    region_bounds (dict): A dictionary specifying the bounds of the region in the format {"min_x": value, "max_x": value, "min_y": value, "max_y": value, "min_z": value, "max_z": value}.

    RETURNS: \n
    new_structured_system (StructuredSystem): A new structured system object with the specified atoms removed.
    '''

    new_structured_system = copy.deepcopy(structured_system)

    for molecule in new_structured_system.molecule_list:
        new_atom_list = []
        for atom in molecule.atoms:
            if (region_bounds["min_x"] <= atom.position[0] <= region_bounds["max_x"] and
                region_bounds["min_y"] <= atom.position[1] <= region_bounds["max_y"] and
                region_bounds["min_z"] <= atom.position[2] <= region_bounds["max_z"]):
                new_atom_list.append(atom)
        molecule.atoms = new_atom_list

    new_structured_system.box_dimensions["min_x"] -= buffer
    new_structured_system.box_dimensions["max_x"] += buffer
    new_structured_system.box_dimensions["min_y"] -= buffer
    new_structured_system.box_dimensions["max_y"] += buffer
    new_structured_system.box_dimensions["min_z"] -= buffer
    new_structured_system.box_dimensions["max_z"] += buffer


    return new_structured_system

def read_molecular_data_json_entry(alias_key : str, file_path = MOLECULAR_DATA_JSON):
    '''
    '''
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    
    molecule_name = json_data["aliases"][alias_key]

    return json_data["molecules"][molecule_name]

def get_alias_key_from_file_path(file_path):
    '''
    '''

    return file_path.split("/")[-1].split(".")[0]


def get_lammps_data_file_indexes(file_path)->dict:
    '''
    '''

    atoms_start = 0
    atoms_stop = 0
    bonds_start = 0
    bonds_stop = 0
    angles_start = 0
    angles_stop = 0
    masses_start = 0
    masses_stop = 0

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            atoms_count_id = next((i for i, line in enumerate(lines) if line.endswith("atoms\n")), float('inf'))
            bonds_count_id = next((i for i, line in enumerate(lines) if line.endswith("bonds\n")), float('inf'))
            angles_count_id = next((i for i, line in enumerate(lines) if line.endswith("angles\n")), float('inf'))
            
            atoms_start = next((i for i, line in enumerate(lines) if line.startswith("Atoms")), float('inf')) + 2
            atoms_stop = int(lines[atoms_count_id].split(" ")[0]) + atoms_start - 1

            bonds_start = next((i for i, line in enumerate(lines) if line.startswith("Bonds")), float('inf')) + 2
            bonds_stop = int(lines[bonds_count_id].split(" ")[0]) + bonds_start - 1

            angles_start = next((i for i, line in enumerate(lines) if line.startswith("Angles")), float('inf')) + 2
            angles_stop = int(lines[angles_count_id].split(" ")[0]) + angles_start - 1

            masses_start = next((i for i, line in enumerate(lines) if line.endswith("zlo zhi\n")), float('inf')) + 4
            masses_stop = next((i for i, line in enumerate(lines) if line.startswith("Atoms")), float('inf')) - 2

            box_dims_start = next((i for i, line in enumerate(lines) if line.endswith("xlo xhi\n")), float('inf'))
            box_dims_stop = next((i for i, line in enumerate(lines) if line.endswith("zlo zhi\n")), float('inf'))
            
    except Exception as e:
        print(f"Error in reading LAMMPS Data File: {e}")

    return {"atoms" : (atoms_start, atoms_stop), "bonds" : (bonds_start, bonds_stop), "angles" : (angles_start, angles_stop), "masses" : (masses_start, masses_stop), "box_dims" : (box_dims_start, box_dims_stop)}


def get_lammps_dump_file_indices(file_path):
    '''
    '''

    timestep_str = "ITEM: TIMESTEP"
    atom_count_str = "ITEM: NUMBER OF ATOMS"
    box_bounds_str = "ITEM: BOX BOUNDS"
    atoms_str = "ITEM: ATOMS"

    with open(file_path, "r") as f:
        lines = f.readlines()

    atom_counts_idxs = []
    atoms_start_idxs = []
    box_bounds_start_idxs = []
    timestep_idxs = []

    for i in range(len(lines)):
        if lines[i].startswith(timestep_str):
            timestep_idxs.append(i+1)
        elif lines[i].startswith(atom_count_str):
            atom_counts_idxs.append(i+1)
        elif lines[i].startswith(box_bounds_str):
            box_bounds_start_idxs.append(i+1)
        elif lines[i].startswith(atoms_str):
            atoms_start_idxs.append(i+1)

    return {
        "atom_counts" : atom_counts_idxs,
        "atoms" : atoms_start_idxs,
        "box_bounds" : box_bounds_start_idxs,
        "timesteps" : timestep_idxs
    }

