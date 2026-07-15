import numpy as np
import math
import subprocess
import random
import os
from MDToolkit.utils.structure_file_utils import estimate_number_density, estimate_solute_solvent_number_density_from_molarity, molecular_formula_to_molar_mass, molecular_formula_to_elements_and_stoichiometries
from MDToolkit.IO.read_file import packmol_pdb_file_to_frame
from MDToolkit.paths import PDB_FILES, XYZ_FILES, PACKMOL_INPUT_FILES, OUTPUT

def create_water_box(box_dimensions: dict, H2O_geometry = None, packmol_helper_file_name = "H2O_box_packmol_helper.inp", packmol_helper_path = PACKMOL_INPUT_FILES, water_box_output_file_name = "H2O_box.pdb", seed = None, density_correction = 0):
    '''
    '''

    if seed is None:
        seed = random.randint(1, 100000)

    if H2O_geometry is not None:
        H2O_pbd_file_path = os.path.join(PDB_FILES, "H2O_" + H2O_geometry + ".pdb")
        print("H2O Template File: " + H2O_pbd_file_path)
    else:
        H2O_pbd_file_path=os.path.join(PDB_FILES, "H2O.pdb")
        print("H2O Template File: " + H2O_pbd_file_path)

    x_len = box_dimensions["max_x"] - box_dimensions["min_x"]

    y_len = box_dimensions["max_y"] - box_dimensions["min_y"]

    z_len = box_dimensions["max_z"] - box_dimensions["min_z"]

    H2O_system = packmol_pdb_file_to_frame(H2O_pbd_file_path)

    H2O_system.populate_elemental_properties_for_all_atoms()

    molecular_weight_H2O = sum([
        atom.elemental_properties["AtomicMass"]
        for molecule in H2O_system.molecule_list
        for atom in molecule.atoms
    ])

    # density = 1.0 g/cm^3
    num_molecules = math.floor((estimate_number_density(density=1.0, molecular_weight=molecular_weight_H2O) * x_len * y_len * z_len) * (1 + density_correction))

    tolerance = 2.0 # default
    output_file_name = water_box_output_file_name
    output_file_path = OUTPUT

    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    if not os.path.exists(packmol_helper_path):
        os.makedirs(packmol_helper_path)

    full_helper_file_path = os.path.join(packmol_helper_path, packmol_helper_file_name)
    full_output_file_path = os.path.join(output_file_path, output_file_name)

    try:
        with open(full_helper_file_path, 'w') as f:
            f.write("# General parameters\n")
            f.write(f"tolerance {tolerance}\n")
            f.write("filetype pdb\n")
            f.write(f"seed {seed}\n")
            f.write(f"output {full_output_file_path}\n\n")

            f.write("# Liquid molecule\n")
            f.write(f"structure {H2O_pbd_file_path}\n")
            f.write(f"\tnumber {num_molecules}\n")
            f.write(f"\tinside box {box_dimensions['min_x']} {box_dimensions['min_y']} {box_dimensions['min_z']} {box_dimensions['max_x']} {box_dimensions['max_y']} {box_dimensions['max_z']}\n")
            f.write("end structure")
        print(f"Packmol input file successfully written to {full_helper_file_path}")
    except Exception as e:
        print(f"Error occurred while writing packmol input file: {e}")

    packmol_helper_full_path = os.path.join(packmol_helper_path, packmol_helper_file_name)

    subprocess.run(f"packmol < {packmol_helper_full_path}", shell = True)

    with open(packmol_helper_full_path, 'r') as f:
            for line in f:
                stripped_line = line.strip()

                if stripped_line.startswith("output"):
                    output_path = stripped_line.split(maxsplit=1)[1]
                    break

    try:
        water_box_system = packmol_pdb_file_to_frame(output_path)
    except Exception as e:
        print(f"Error occurred while reading packmol output file: {e}")

    return water_box_system

# def create_ionic_liquid_box(box_dimensions : dict, cation_pdb_file_path : str, anion_pdb_file_path : str, ionic_liquid_density : float = 1.0, packmol_helper_file_name = "ionic_liquid_box_packmol_helper.inp", packmol_helper_path = PACKMOL_INPUT_FILES,
#                             ionic_liquid_output_file_path = OUTPUT, ionic_liquid_output_file_name : str = "ionic_liquid_box.pdb"):
#     """
#     """
#     x_len = box_dimensions["max_x"] - box_dimensions["min_x"]

#     y_len = box_dimensions["max_y"] - box_dimensions["min_y"]

#     z_len = box_dimensions["max_z"] - box_dimensions["min_z"]

#     cation_system = pdb_file_to_structured_system(cation_pdb_file_path)
#     anion_system = pdb_file_to_structured_system(anion_pdb_file_path)

#     cation_system.populate_elemental_properties_for_all_atoms()
#     anion_system.populate_elemental_properties_for_all_atoms()

#     molecular_weight_cation = sum([
#          atom.elemental_properties["AtomicMass"]
#          for molecule in cation_system.molecule_list
#          for atom in molecule.atoms
#     ])

#     molecular_weight_anion = sum([
#          atom.elemental_properties["AtomicMass"]
#          for molecule in anion_system.molecule_list
#          for atom in molecule.atoms
#     ])

#     num_molecules = math.floor(estimate_number_density(density=ionic_liquid_density, molecular_weight=molecular_weight_cation + molecular_weight_anion) * x_len * y_len * z_len)

#     tolerance = 2.0 # Typical cutoff

#     if not os.path.exists(packmol_helper_path):
#         os.makedirs(packmol_helper_path)

#     full_helper_file_path = os.path.join(packmol_helper_path, packmol_helper_file_name)
#     full_output_file_path = os.path.join(ionic_liquid_output_file_path, ionic_liquid_output_file_name)

#     try:
#         with open(full_helper_file_path, 'w') as f:
#             f.write("# General parameters\n")
#             f.write(f"tolerance {tolerance}\n")
#             f.write("filetype pdb\n")
#             f.write(f"output {full_output_file_path}\n\n")

#             f.write("# Cation\n")
#             f.write(f"structure {cation_pdb_file_path}\n")
#             f.write(f"\tnumber {num_molecules}\n")
#             f.write(f"\tinside box {box_dimensions['min_x']} {box_dimensions['min_y']} {box_dimensions['min_z']} {box_dimensions['max_x']} {box_dimensions['max_y']} {box_dimensions['max_z']}\n")
#             f.write("end structure\n\n")

#             f.write("# Anion\n")
#             f.write(f"structure {anion_pdb_file_path}\n")
#             f.write(f"\tnumber {num_molecules}\n")
#             f.write(f"\tinside box {box_dimensions['min_x']} {box_dimensions['min_y']} {box_dimensions['min_z']} {box_dimensions['max_x']} {box_dimensions['max_y']} {box_dimensions['max_z']}\n")
#             f.write("end structure")
#         print(f"Packmol input file successfully written to {full_helper_file_path}")
#     except Exception as e:
#         print(f"Error occurred while writing packmol input file: {e}")

#     packmol_helper_full_path = os.path.join(packmol_helper_path, packmol_helper_file_name)

#     subprocess.run(f"packmol < {packmol_helper_full_path}", shell = True)

#     with open(packmol_helper_full_path, 'r') as f:
#             for line in f:
#                 stripped_line = line.strip()

#                 if stripped_line.startswith("output"):
#                     output_path = stripped_line.split(maxsplit=1)[1]
#                     break
#     try:
#         ionic_liquid_box_system = packmol_pdb_file_to_frame(output_path)
#     except Exception as e:
#         print(f"Error occurred while reading packmol output file: {e}")

#     return ionic_liquid_box_system

# def create_simplesalt_in_water_box(box_dimensions: dict, salt_molecular_formula: str, salt_molarity: float, solution_density = 1.0, H2O_geometry = None, packmol_helper_file_name = "H2O_Salt_box_packmol_helper.inp", packmol_helper_path = PACKMOL_INPUT_FILES, water_box_output_file_name = "Salt_H2O_box.pdb", seed = None, density_correction = 0):
#     '''
#     '''

#     if seed is None:
#             seed = random.randint(1, 100000)

#     if H2O_geometry is not None:
#         H2O_pbd_file_path = os.path.join(PDB_FILES, "H2O_" + H2O_geometry + ".pdb")
#         print("H2O Template File: " + H2O_pbd_file_path)
#     else:
#         H2O_pbd_file_path=os.path.join(PDB_FILES, "H2O.pdb")
#         print("H2O Template File: " + H2O_pbd_file_path)

#     x_len = box_dimensions["max_x"] - box_dimensions["min_x"]

#     y_len = box_dimensions["max_y"] - box_dimensions["min_y"]

#     z_len = box_dimensions["max_z"] - box_dimensions["min_z"]

#     try:
#         H2O_system = pdb_file_to_structured_system(H2O_pbd_file_path)
#     except:
#         H2O_system = packmol_pdb_file_to_structured_system(H2O_pbd_file_path)

#     H2O_system.populate_elemental_properties_for_all_atoms()

#     molecular_weight_H2O = sum([
#         atom.elemental_properties["AtomicMass"]
#         for molecule in H2O_system.molecule_list
#         for atom in molecule.atoms
#     ])

#     molecular_weight_salt = molecular_formula_to_molar_mass(salt_molecular_formula)

#     salt_elems, salt_stoichs = molecular_formula_to_elements_and_stoichiometries(salt_molecular_formula)

#     # density = 1.0 g/cm^3
#     num_molecules_solv, num_molecules_solute = estimate_solute_solvent_number_density_from_molarity(salt_molarity, molecular_weight_H2O, molecular_weight_salt, solution_density)

#     num_molecules_solv = math.floor((num_molecules_solv * x_len * y_len * z_len) * (1 + density_correction))
#     num_molecules_solute = math.floor((num_molecules_solute * x_len * y_len * z_len) * (1 + density_correction))

#     tolerance = 2.0 # default
#     output_file_name = water_box_output_file_name
#     output_file_path = OUTPUT

#     if not os.path.exists(output_file_path):
#         os.makedirs(output_file_path)

#     if not os.path.exists(packmol_helper_path):
#         os.makedirs(packmol_helper_path)

#     full_helper_file_path = os.path.join(packmol_helper_path, packmol_helper_file_name)
#     full_output_file_path = os.path.join(output_file_path, output_file_name)

#     try:
#         with open(full_helper_file_path, 'w') as f:
#             f.write("# General parameters\n")
#             f.write(f"tolerance {tolerance}\n")
#             f.write("filetype pdb\n")
#             f.write(f"seed {seed}\n")
#             f.write(f"output {full_output_file_path}\n\n")

#             f.write("# Liquid molecule\n")
#             f.write(f"structure {H2O_pbd_file_path}\n")
#             f.write(f"\tnumber {num_molecules_solv}\n")
#             f.write(f"\tinside box {box_dimensions['min_x']} {box_dimensions['min_y']} {box_dimensions['min_z']} {box_dimensions['max_x']} {box_dimensions['max_y']} {box_dimensions['max_z']}\n")
#             f.write("end structure\n\n")

#             for i in range(len(salt_elems)):
#                 f.write(f"# Salt Spcs: {i+ 1}\n")
#                 f.write(f"structure {os.path.join(PDB_FILES, salt_elems[i] + '.pdb')}\n")
#                 f.write(f"\tnumber {int(num_molecules_solute * salt_stoichs[i])}\n")
#                 f.write(f"\tinside box {box_dimensions['min_x']} {box_dimensions['min_y']} {box_dimensions['min_z']} {box_dimensions['max_x']} {box_dimensions['max_y']} {box_dimensions['max_z']}\n")
#                 f.write("end structure\n\n")
            
#         print(f"Packmol input file successfully written to {full_helper_file_path}")
#     except Exception as e:
#         print(f"Error occurred while writing packmol input file: {e}")

#     packmol_helper_full_path = os.path.join(packmol_helper_path, packmol_helper_file_name)

#     subprocess.run(f"packmol < {packmol_helper_full_path}", shell = True)

#     with open(packmol_helper_full_path, 'r') as f:
#             for line in f:
#                 stripped_line = line.strip()

#                 if stripped_line.startswith("output"):
#                     output_path = stripped_line.split(maxsplit=1)[1]
#                     break

#     try:
#         system = packmol_pdb_file_to_frame(output_path)
#     except Exception as e:
#         print(f"Error occurred while reading packmol output file: {e}")

#     return system