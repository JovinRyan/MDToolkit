import numpy as np
import math
import subprocess
import os
from MDToolkit.utils.structure_file_utils import estimate_number_density
from MDToolkit.IO.read_file import pdb_file_to_structured_system, packmol_pdb_file_to_structured_system
from MDToolkit.IO.write_file import write_liquid_packmol_helper_file
from MDToolkit.data.objects import StructuredSystem, Molecule, Atom

def create_water_box(box_dimensions: dict, H2O_pbd_file_path="/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/common_pdb_files/H2O.pdb", packmol_helper_file_name = "H2O_box_packmol_helper.inp", packmol_helper_path = "/home/jovinryanj/projects/mdtoolkit/packmol_input_files"):
    '''
    '''

    x_len = box_dimensions["max_x"] - box_dimensions["min_x"]

    y_len = box_dimensions["max_y"] - box_dimensions["min_y"]

    z_len = box_dimensions["max_z"] - box_dimensions["min_z"]

    H2O_system = pdb_file_to_structured_system(H2O_pbd_file_path)

    H2O_system.populate_elemental_properties_for_all_atoms()

    molecular_weight_H2O = sum([
        atom.elemental_properties["AtomicMass"]
        for molecule in H2O_system.molecule_list
        for atom in molecule.atoms
    ])

    # density = 1.0 g/cm^3
    num_molecules = math.floor(estimate_number_density(density=1.0, molecular_weight=molecular_weight_H2O) * x_len * y_len * z_len)

    # water vdw radius = 1.4 A
    write_liquid_packmol_helper_file(H2O_pbd_file_path, box_dimensions, num_molecules, output_file_name = "H2O_box.pdb", tolerance = 1.4, packmol_helper_file_name = packmol_helper_file_name, packmol_helper_path = packmol_helper_path)

    packmol_helper_full_path = os.path.join(packmol_helper_path, packmol_helper_file_name)

    subprocess.run(f"packmol < {packmol_helper_full_path}", shell = True)

    with open(packmol_helper_full_path, 'r') as f:
            for line in f:
                stripped_line = line.strip()

                if stripped_line.startswith("output"):
                    output_path = stripped_line.split(maxsplit=1)[1]
                    break
    water_box_system = packmol_pdb_file_to_structured_system(output_path)

    return water_box_system
