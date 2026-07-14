from pathlib import Path
import numpy as np
from MDToolkit.data.objects import Frame, Topology
from MDToolkit.data.misc_objects import BoxVolume
from MDToolkit.utils.structure_file_utils import create_elements_dictionary

def lammps_data_file_to_topology(filepath, elements_dict : dict = None):
    '''
    '''
    if elements_dict is None:
        elements_dict = create_elements_dictionary()
    
    mass_lines = []
    with open(filepath, "r", encoding = "ascii") as f:
        in_masses = False
        for line in f:
            if line.strip() == "Masses":
                in_masses = True
                next(f) 
                continue
            if in_masses:
                if not line.strip():
                    break

                mass_lines.append(line.rstrip("\n"))

    mass_mapping = {int(line.split(" ")[0]) : float(line.split(" ")[1]) for line in mass_lines}

    type_mapping = {
        i : min(
            elements_dict.keys(),
            key = lambda element: abs(
                elements_dict[element]["AtomicMass"] - mass_mapping[i]
            )
        )
        for i in mass_mapping.keys()
    }

    return Topology(type_mapping, elements_dict)

def lammps_data_file_to_frame(filepath : Path, topology : Topology = None, elements_dict = create_elements_dictionary()):
    '''
    '''

    atom_styles = {
        "atomic" : ["id", "type", "x", "y", "z"],
        "charge" : ["id", "type", "q", "x", "y", "z"],
        "full" : ["id", "mol", "type", "q", "x", "y", "z"]
    }

    if topology is None:
        topology = lammps_data_file_to_topology(filepath, elements_dict)

    frame = Frame(topology)

    type_to_charge = {}

    with open(filepath, "r") as f:

        in_atoms = False
        atom_idx = 0

        for line in f:

            line = line.strip()

            if not line:
                continue

            if line.endswith(" atoms"):

                frame.num_atoms = int(line.split()[0])

                frame.ids = np.empty(frame.num_atoms, dtype = np.int32)
                frame.types = np.empty(frame.num_atoms, dtype = np.int32)
                frame.positions = np.empty((frame.num_atoms, 3), dtype = np.float64)

            elif line.endswith(" xlo xhi"):

                xlo, xhi = map(float, line.split()[:2])

            elif line.endswith(" ylo yhi"):

                ylo, yhi = map(float, line.split()[:2])

            elif line.endswith(" zlo zhi"):

                zlo, zhi = map(float, line.split()[:2])

                frame.box = BoxVolume(
                    [xlo, ylo, zlo],
                    [xhi, yhi, zhi]
                )

            elif line.startswith("Atoms"):

                atom_style = line.split("#")[1].strip()

                if atom_style not in atom_styles:
                    raise ValueError(f"Unsupported atom style '{atom_style}'")

                column_map = {
                    name : i
                    for i, name in enumerate(atom_styles[atom_style])
                }

                if "mol" in column_map:
                    frame.mol_ids = np.empty(frame.num_atoms, dtype = np.int32)

                in_atoms = True

                next(f)

            elif in_atoms:

                if line[0].isalpha():
                    break

                fields = line.split()

                atom_type = int(fields[column_map["type"]])

                frame.ids[atom_idx] = int(fields[column_map["id"]])
                frame.types[atom_idx] = atom_type

                if frame.mol_ids is not None:
                    frame.mol_ids[atom_idx] = int(fields[column_map["mol"]])

                frame.positions[atom_idx, 0] = float(fields[column_map["x"]])
                frame.positions[atom_idx, 1] = float(fields[column_map["y"]])
                frame.positions[atom_idx, 2] = float(fields[column_map["z"]])

                if "q" in column_map and atom_type not in type_to_charge:
                    type_to_charge[atom_type] = float(fields[column_map["q"]])

                atom_idx += 1

    order = np.argsort(frame.ids)

    frame.ids = frame.ids[order]
    frame.types = frame.types[order]
    frame.positions = frame.positions[order]

    if frame.mol_ids is not None:
        frame.mol_ids = frame.mol_ids[order]
    
    frame.topology.rewrite_charges(type_to_charge)

    return frame

def packmol_pdb_file_to_frame(filepath: Path, topology: Topology = None, elements_dict=create_elements_dictionary()):
    '''
    '''

    atom_data = []

    with open(filepath, "r", encoding="ascii") as f:
        for line in f:

            if not line.startswith(("ATOM", "HETATM")):
                continue

            atom_data.append({
                "id": int(line[6:11]),
                "mol": int(line[22:26]),
                "element": line[76:78].strip(),
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
            })

    if topology is None:

        unique_elements = sorted({
            atom["element"]
            for atom in atom_data
        })

        type_mapping = {
            i + 1: element
            for i, element in enumerate(unique_elements)
        }

        topology = Topology(type_mapping, elements_dict)

    element_to_type = {
        element: atom_type
        for atom_type, element in topology.type_mapping.items()
    }

    frame = Frame(topology)

    frame.num_atoms = len(atom_data)

    frame.ids = np.empty(frame.num_atoms, dtype=np.int32)
    frame.types = np.empty(frame.num_atoms, dtype=np.int32)
    frame.mol_ids = np.empty(frame.num_atoms, dtype=np.int32)
    frame.positions = np.empty((frame.num_atoms, 3), dtype=np.float64)

    for i, atom in enumerate(atom_data):

        frame.ids[i] = atom["id"]
        frame.types[i] = element_to_type[atom["element"]]
        frame.mol_ids[i] = atom["mol"]

        frame.positions[i, 0] = atom["x"]
        frame.positions[i, 1] = atom["y"]
        frame.positions[i, 2] = atom["z"]

    order = np.argsort(frame.ids)

    frame.ids = frame.ids[order]
    frame.types = frame.types[order]
    frame.mol_ids = frame.mol_ids[order]
    frame.positions = frame.positions[order]

    return frame

def cif_file_to_frame(file_path : Path, topology : Topology = None, elements_dict = create_elements_dictionary()):
    '''
    '''

    with open(file_path, "r") as file:
        for line in file.readlines():
            if line.startswith("_cell_length_a"):
                min_x = 0
                max_x = float(line.split(" ")[-1])
            if line.startswith("_cell_length_b"):
                min_y = 0
                max_y = float(line.split(" ")[-1])
            if line.startswith("_cell_length_c"):
                min_z = 0
                max_z = float(line.split(" ")[-1])
    

    box = BoxVolume([min_x, min_y, min_z], [max_x, max_y, max_z])