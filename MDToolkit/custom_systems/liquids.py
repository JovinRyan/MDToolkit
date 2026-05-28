import numpy as np
import math
from MDToolkit.utils.structure_file_utils import estimate_number_density
from MDToolkit.IO.read_file import pdb_file_to_structured_system
from MDToolkit.data.objects import StructuredSystem, Molecule, Atom


def create_water_box(box_dimensions: dict, edge_buffers=[1.0, 1.0, 1.0], H2O_pbd_file_path="/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/common_pdb_files/H2O.pdb"):
    '''
    '''

    x_len = (box_dimensions["max_x"] - box_dimensions["min_x"] - edge_buffers[0])

    y_len = (box_dimensions["max_y"] - box_dimensions["min_y"] - edge_buffers[1])

    z_len = (box_dimensions["max_z"] - box_dimensions["min_z"] - edge_buffers[2])

    H2O_system = pdb_file_to_structured_system(H2O_pbd_file_path)

    H2O_system.populate_elemental_properties_for_all_atoms()

    H2O_system.set_COM_to_origin()

    molecular_weight_H2O = sum([
        atom.elemental_properties["AtomicMass"]
        for molecule in H2O_system.molecule_list
        for atom in molecule.atoms
    ])

    num_molecules = math.floor(estimate_number_density(density=1.0, molecular_weight=molecular_weight_H2O) * x_len * y_len * z_len)

    print(num_molecules)

    # Estimate cubic spacing
    volume_per_molecule = (x_len * y_len * z_len) / num_molecules

    spacing = volume_per_molecule ** (1/3)

    nx = max(1, int(x_len / spacing))
    ny = max(1, int(y_len / spacing))
    nz = max(1, int(z_len / spacing))

    xs = np.linspace(
        box_dimensions["min_x"] + edge_buffers[0]/2,
        box_dimensions["max_x"] - edge_buffers[0]/2,
        nx
    )

    ys = np.linspace(
        box_dimensions["min_y"] + edge_buffers[1]/2,
        box_dimensions["max_y"] - edge_buffers[1]/2,
        ny
    )

    zs = np.linspace(
        box_dimensions["min_z"] + edge_buffers[2]/2,
        box_dimensions["max_z"] - edge_buffers[2]/2,
        nz
    )

    placed_molecules = []

    molecule_id = 1
    atom_id = 1

    count = 0

    for x in xs:
        for y in ys:
            for z in zs:

                if count >= num_molecules:
                    break

                new_molecule = Molecule(molecule_id=molecule_id, molecule_name=H2O_system.molecule_list[0].name,  atoms = H2O_system.molecule_list[0].atoms)

                for atom in new_molecule.atoms:

                    atom.position = [
                        atom.position[0] + x,
                        atom.position[1] + y,
                        atom.position[2] + z
                    ]

                    atom.id = atom_id

                    atom_id += 1

                placed_molecules.append(
                    new_molecule
                )

                molecule_id += 1
                count += 1

    water_system = StructuredSystem(
        molecule_list=placed_molecules,
        box_dimensions=box_dimensions
    )

    return water_system
