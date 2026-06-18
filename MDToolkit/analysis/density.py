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

        total_mass = sum(
            atom.elemental_properties["AtomicMass"]
            for atom in atom_list
        )

        box_volume = (
            (system.box_dimensions["max_x"] - system.box_dimensions["min_x"]) *
            (system.box_dimensions["max_y"] - system.box_dimensions["min_y"]) *
            (system.box_dimensions["max_z"] - system.box_dimensions["min_z"])
        )

        avg_density = total_mass * 1e24 / (sc.N_A * box_volume)
        
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

    elemental_number_density = {element: np.zeros(bins) for element in element_set}

    for bin_idx, entry in enumerate(elements_in_bin):
        for element, value in entry.items():
            elemental_number_density[element][bin_idx] = value

    return {
        "bin_edges" : bin_edges,
        "bin_centers": 0.5 * (bin_edges[:-1] + bin_edges[1:]),
        "bin_volumes": bin_volumes,
        "number_density": atoms_in_bin / total_atom_number,
        "elemental_number_density" : elemental_number_density,
        "density" : density_in_bin,
        "average_density" : avg_density
    }

def averaged_axial_density(simulation, axis = "x", volume_method = "box", bins = 200, discrete_volume = None):
    '''
    Computes the axial density profile averaged over all frames in a simulation.\n
    Parameters:
    - simulation (Simulation): The Simulation object containing the frames to analyze.
    - axis (str): The axis along which to compute the density profile ('x', 'y', or 'z').
    - volume_method (str): The method to use for calculating the volume of each bin ('box' or 'discrete').
    - bins (int): The number of bins to use for the density profile.
    - discrete_volume (dict): A dictionary containing the discrete volume for each bin if volume_method is 'discrete'. The keys should be bin indices and the values should be the corresponding volumes.
    - return_std_dev (bool): Whether to return the standard deviation for the density profile.\n
    Returns:
    - 'bin_edges': The edges of the bins used for the density profile.
    - 'bin_centers': The centers of the bins used for the density profile.
    - 'bin_volumes': The volumes of each bin used for the density profile.
    - 'number_density_mean': The mean number density profile averaged over all frames.
    - 'number_density_std': The standard deviation or standard error of the mean for the number density profile, depending on the return_std_err and return_std_dev parameters.
    - 'density_mean': The mean mass density profile averaged over all frames.
    - 'density_std': The standard deviation or standard error of the mean for the mass density profile, depending on the return_std_err and return_std_dev parameters.
    - 'elemental_number_density_mean': A dictionary containing the mean elemental number density profiles for each element, averaged over all frames. The keys are the element symbols and the values are arrays of the mean elemental number density for each bin.
    - 'elemental_number_density_std': A dictionary containing the standard deviation or standard error of the mean for the elemental number density profiles for
    '''

    number_density_stack = []
    density_stack = []
    elemental_stack = []
    average_density_list = []

    bin_edges = None
    bin_centers = None
    bin_volumes = None
    elemental_keys = None

    for frame in simulation.frames:

        result = axial_density(frame, axis = axis, volume_method = volume_method, bins = bins, discrete_volume = discrete_volume)

        if bin_edges is None:
            bin_edges = result["bin_edges"]
            bin_centers = result["bin_centers"]
            bin_volumes = result["bin_volumes"]
            elemental_keys = list(result["elemental_number_density"].keys())

        number_density_stack.append(result["number_density"])
        density_stack.append(result["density"])
        elemental_stack.append(result["elemental_number_density"])
        average_density_list.append(result["average_density"])

    number_density_stack = np.array(number_density_stack)
    density_stack = np.array(density_stack)

    number_density_mean = np.mean(number_density_stack, axis = 0)
    density_mean = np.mean(density_stack, axis = 0)

    number_density_std = np.std(number_density_stack, axis=0)

    density_std = np.std(density_stack, axis=0)

    elemental_number_density_mean = {element : np.zeros(bins) for element in elemental_keys}
    elemental_number_density_std = {element : np.zeros(bins) for element in elemental_keys}

    average_density = np.mean(average_density_list)

    average_density_std = np.std(average_density_list)

    for element in elemental_keys:

        temp = np.array([
            frame[element][i] if element in frame else 0.0
            for frame in elemental_stack
            for i in range(bins)
        ]).reshape(len(simulation.frames), bins)

        elemental_number_density_mean[element] = np.mean(temp, axis = 0)
        elemental_number_density_std[element] = np.std(temp, axis = 0)

    return {
        "bin_edges" : bin_edges,
        "bin_centers" : bin_centers,
        "bin_volumes" : bin_volumes,
        "number_density_mean" : number_density_mean,
        "number_density_std" : number_density_std,
        "density_mean" : density_mean,
        "density_std" : density_std,
        "elemental_number_density_mean" : elemental_number_density_mean,
        "elemental_number_density_std" : elemental_number_density_std,
        "average_density" : average_density,
        "average_density_std" : average_density_std
    }