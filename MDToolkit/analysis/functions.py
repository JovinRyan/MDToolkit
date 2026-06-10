import numpy as np
import scipy.constants as sc
from MDToolkit.data.objects import StructuredSystem, Molecule, Atom
from MDToolkit.utils.misc_utils import sort_atom_list_by_index

def axial_density(system : StructuredSystem, axis = "x", volume_method = "box", bins = 200, discrete_volume = None):
    '''
    '''

    axes_set = {"x", "y", "z"}

    if axis not in axes_set:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    system.populate_elemental_properties_for_all_atoms()

    axis_index = 0

    match axis:
        case "x":
            axis_index = 0
        case "y":
            axis_index = 1
        case "z":
            axis_index = 2

    atom_list = [atom for molecule in system.molecule_list for atom in molecule.atoms]
    atom_index_axial_coordinate_list = [[atom.id, np.float32(atom.position[axis_index])] for molecule in system.molecule_list for atom in molecule.atoms]
    element_set, counts = np.unique([atom.element for atom in atom_list], return_counts = True)

    element_count_dict = dict(zip(element_set, counts))

    sorted_atom_list = sort_atom_list_by_index(atom_list)

    total_atom_number = len(sorted_atom_list)

    unselected_axes = axes_set - {axis}

    axis_dims = (
        system.box_dimensions[f"min_{axis}"],
        system.box_dimensions[f"max_{axis}"]
    )

    bin_edges = np.linspace(axis_dims[0], axis_dims[1], bins + 1)

    unselected_axes_dims = [
        [
            system.box_dimensions[f"min_{ax}"],
            system.box_dimensions[f"max_{ax}"]
        ]
        for ax in unselected_axes
    ]


    if volume_method == "box" and discrete_volume is None:
        atom_ids = np.array([x[0] for x in atom_index_axial_coordinate_list])
        coords = np.array([x[1] for x in atom_index_axial_coordinate_list])

        bin_indices = np.digitize(coords, bin_edges) - 1

        bin_volumes = np.full(bins, ((axis_dims[1] - axis_dims[0]) / bins) * np.prod([dim[1] - dim[0] for dim in unselected_axes_dims]))
        
    atom_ids_in_bin = {i: [] for i in range(bins)}
    atoms_in_bin = {i: [] for i in range(bins)}
    elements_in_bin = {i : [] for i in range(bins)}
    masses_in_bin = {i : [] for i in range(bins)}

    for atom_id, bin_idx in zip(atom_ids, bin_indices):
        if 0 <= bin_idx < bins:
            atom_ids_in_bin[bin_idx].append(atom_id)
    
    for bin_idx in range(bins):
        atoms = [
        sorted_atom_list[atom_id - 1]
        for atom_id in atom_ids_in_bin[bin_idx]
        ]
        elements = [
            sorted_atom_list[atom_id - 1].element
            for atom_id in atom_ids_in_bin[bin_idx]
        ]
        elements_list, counts = np.unique(elements, return_counts=True)
        elements_in_bin[bin_idx] = dict(zip(elements_list, counts))
        
        atoms_in_bin[bin_idx] = sum(counts)

        masses_in_bin[bin_idx] = sum(atom.elemental_properties["AtomicMass"] for atom in atoms)

    atoms_in_bin = np.array([atoms_in_bin[i]for i in range(bins)], dtype=float)

    elements_in_bin = [elements_in_bin[i] for i in range(len(elements_in_bin))]

    for entry in elements_in_bin:
        for key in entry:
            entry[key] = entry[key] / element_count_dict[key]
    
    density_in_bin = np.array([masses_in_bin[i] * 1e24 / (sc.N_A * bin_volumes[i])for i in range(bins)])

    return {
        "bin_edges" : bin_edges,
        "bin_centers": 0.5 * (bin_edges[:-1] + bin_edges[1:]),
        "bin_volumes": bin_volumes,
        "number_density": atoms_in_bin / total_atom_number,
        "elemental_number_density" : elements_in_bin,
        "density" : density_in_bin
    }