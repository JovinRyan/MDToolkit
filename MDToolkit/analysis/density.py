import numpy as np
import os
import scipy.constants as sc
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from MDToolkit.data.objects import Simulation, StructuredSystem, Molecule, Atom
from MDToolkit.utils.misc_utils import sort_atom_list_by_index, get_n_even_chunks
from MDToolkit.data.misc_objects import Volume, BoxVolume, CylinderVolume

def axial_density(system : StructuredSystem, volumes: Sequence[Volume], bins = 250, axis = "x") -> dict:
    '''
    '''

    axes_set = {"x", "y", "z"}

    if axis not in axes_set:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    axis_index = 0

    match axis:
        case "x":
            axis_index = 0
        case "y":
            axis_index = 1
        case "z":
            axis_index = 2
    
    system.populate_elemental_properties_for_all_atoms()

    atoms_list = system.get_atoms_list()
    axial_coordinates_list = [atom.position[axis_index] for atom in atoms_list]

    vol_points = []

    for vol in volumes:
        vol_points.append((vol.point1, vol.point2))

    axis_points = []
    for i in range(len(vol_points)):
        axis_points.append(vol_points[i][0][axis_index])
        axis_points.append(vol_points[i][1][axis_index])

    axis_bins = np.linspace(min(axis_points), max(axis_points), num = bins + 1)

    atoms_bin_idxs = np.digitize(axial_coordinates_list, axis_bins) - 1

    atoms_bins = [[] for _ in range(bins)]

    for atom, b in zip(atoms_list, atoms_bin_idxs):
        if 0 <= b < bins:
            atoms_bins[b].append(atom)

    bin_ranges = [
        (axis_bins[i], axis_bins[i + 1])
        for i in range(bins)
    ]

    vol_ranges = [
        (
            min(vol.point1[axis_index], vol.point2[axis_index]),
            max(vol.point1[axis_index], vol.point2[axis_index])
        )
        for vol in volumes
    ]

    def overlap(a, b):
        return not (a[1] <= b[0] or b[1] <= a[0])

    bin_to_volumes = [[] for _ in range(bins)]

    for i, b in enumerate(bin_ranges):
        for j, v in enumerate(vol_ranges):
            if overlap(b, v):
                bin_to_volumes[i].append(j)

    atoms_in_volume_bins = [[[] for _ in range(len(volumes))] for _ in range(bins)]

    for i, bin_atoms in enumerate(atoms_bins):
        if len(bin_atoms) == 0:
            continue

        for v_idx in bin_to_volumes[i]:

            vol = volumes[v_idx]

            mask = vol.contains_atoms(bin_atoms)

            filtered_atoms = [atom for atom, m in zip(bin_atoms, mask) if m]

            atoms_in_volume_bins[i][v_idx] = filtered_atoms

    atoms_in_bin = np.array([sum(len(atoms_in_volume_bins[i][v]) for v in range(len(volumes))) for i in range(bins)])

    total_atom_number = sum(atoms_in_bin)

    elements_in_bin = []
    for i in range(bins):

        elements = [
            atom.element
            for v in range(len(volumes))
            for atom in atoms_in_volume_bins[i][v]
        ]

        elems, counts = np.unique(elements, return_counts=True)

        elements_in_bin.append(
            dict(zip(elems, counts))
        )
    
        bin_volumes = np.zeros(bins)

    for i in range(bins):

        bin_min = axis_bins[i]
        bin_max = axis_bins[i + 1]

        for v_idx in bin_to_volumes[i]:

            vol = volumes[v_idx]

            vol_min = min(
                vol.point1[axis_index],
                vol.point2[axis_index]
            )

            vol_max = max(
                vol.point1[axis_index],
                vol.point2[axis_index]
            )

            overlap_length = (
                min(bin_max, vol_max)
                - max(bin_min, vol_min)
            )

            if overlap_length <= 0:
                continue

            axial_length = vol_max - vol_min

            bin_volumes[i] += (
                overlap_length
                * vol.volume
                / axial_length
            )
    
    masses_in_bin = np.array([
        sum(atom.elemental_properties["AtomicMass"] for v in range(len(volumes)) for atom in atoms_in_volume_bins[i][v]) for i in range(bins)])

    density_in_bin = np.full(len(masses_in_bin), np.nan)

    mask = bin_volumes > 0
    density_in_bin[mask] = (masses_in_bin[mask] * 1e24 / (sc.N_A * bin_volumes[mask]))

    avg_density = (np.sum(density_in_bin[mask] * bin_volumes[mask]) / np.sum(bin_volumes[mask]))

    all_elements = sorted({atom.element for atom in atoms_list})

    elemental_number_density = {element: np.zeros(bins) for element in all_elements}

    for bin_idx, entry in enumerate(elements_in_bin):
        for element, value in entry.items():
            elemental_number_density[element][bin_idx] = value
    
    for key in elemental_number_density.keys():
        key_sum = np.sum(elemental_number_density[key])
        if key_sum > 0:
            elemental_number_density[key] = elemental_number_density[key] / key_sum

    valid = slice(1, -1)

    return {
        "bin_edges": axis_bins[1:-1],
        "bin_centers": 0.5 * (axis_bins[:-1] + axis_bins[1:])[valid],
        "bin_volumes": bin_volumes[valid],
        "number_density": (atoms_in_bin / total_atom_number)[valid],
        "elemental_number_density": {
            k: v[valid]
            for k, v in elemental_number_density.items()
        },
        "density": density_in_bin[valid],
        "average_density": avg_density
    }

def axial_density_time_averaged(simulation : Simulation, volumes: Sequence[Volume], bins = 250, axis = "x", n_workers = os.cpu_count()//2) -> dict:
    '''
    '''

    with ProcessPoolExecutor(max_workers = n_workers) as executor:
        futures = {
            executor.submit(axial_density, frame, volumes, bins, axis): i
            for i, frame in enumerate(simulation.frames)
        }

        results = [None] * len(simulation.frames)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Performing density calculations:", unit="frame(s)"):
            i = futures[fut]
            results[i] = fut.result()

    density = np.stack([r["density"] for r in results])
    number_density = np.stack([r["number_density"] for r in results])
    average_density = np.array([r["average_density"] for r in results])

    elements = results[0]["elemental_number_density"].keys()

    elemental_density = {element: np.stack([r["elemental_number_density"][element] for r in results]) for element in elements}

    return {
        "bin_edges": results[0]["bin_edges"],
        "bin_centers": results[0]["bin_centers"],
        "bin_volumes": results[0]["bin_volumes"],

        "density_mean": np.mean(density, axis=0),
        "density_std": np.std(density, axis=0),

        "number_density_mean": np.mean(number_density, axis=0),
        "number_density_std": np.std(number_density, axis=0),

        "average_density_mean": np.mean(average_density),
        "average_density_std": np.std(average_density),

        "elemental_number_density_mean": {
            element: np.mean(arr, axis=0)
            for element, arr in elemental_density.items()
        },

        "elemental_number_density_std": {
            element: np.std(arr, axis=0)
            for element, arr in elemental_density.items()
        }
    }

def radial_density(system : StructuredSystem, cyl_volume : CylinderVolume, bins = 250)-> dict:
    '''
    '''

    annuli = cyl_volume.discretize_radial(bins)

    system.populate_elemental_properties_for_all_atoms()

    atoms_list = system.get_atoms_list()

    atoms_in_annulus = []

    for i in range(len(annuli)):
        if i == len(annuli) - 1:
            mask = annuli[i].contains_atoms(atoms_list, outer_radial_bound="closed")
        else:
            mask = annuli[i].contains_atoms(atoms_list)
        
        atoms_in_annulus.append([atom for atom, keep in zip(atoms_list, mask) if keep])
    
    atoms_per_bin = np.array([len(atoms)for atoms in atoms_in_annulus])

    total_atoms = np.sum(atoms_per_bin)

    bin_volumes = np.array([annulus.volume for annulus in annuli])

    masses_in_bin = np.array([sum(atom.elemental_properties["AtomicMass"] for atom in atoms) for atoms in atoms_in_annulus])

    density_in_bin = np.full(bins, np.nan)

    mask = bin_volumes > 0

    density_in_bin[mask] = (masses_in_bin[mask] * 1e24 / (sc.N_A * bin_volumes[mask]))

    average_density = np.sum(density_in_bin[mask] * bin_volumes[mask]) / np.sum(bin_volumes[mask])

    elements_in_bin = []

    for atoms in atoms_in_annulus:

        elements = [atom.element for atom in atoms]

        elems, counts = np.unique(
            elements,
            return_counts=True
        )

        elements_in_bin.append(
            dict(zip(elems, counts))
        )

    all_elements = sorted({
        atom.element
        for atom in atoms_list
    })

    elemental_number_density = {
        e: np.zeros(bins)
        for e in all_elements
    }

    for i, entry in enumerate(elements_in_bin):
        for element, count in entry.items():
            elemental_number_density[element][i] = count

    for element in elemental_number_density:
        total = elemental_number_density[element].sum()
        if total > 0:
            elemental_number_density[element] /= total

    bin_edges = np.array([annuli[0].i_radius] + [a.o_radius for a in annuli])

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    number_density = atoms_per_bin / total_atoms

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "bin_volumes": bin_volumes,
        "number_density": number_density,
        "elemental_number_density": {
            k: v
            for k, v in elemental_number_density.items()
        },
        "density": density_in_bin,
        "average_density": average_density,
    }

def radial_density_time_averaged(simulation : Simulation, cyl_volume: CylinderVolume, bins = 250, n_workers = os.cpu_count()//2, averaging_blocks = 10) -> dict:
    '''
    '''

    with ProcessPoolExecutor(max_workers = n_workers) as executor:
        futures = {
            executor.submit(radial_density, frame, cyl_volume, bins): i
            for i, frame in enumerate(simulation.frames)
        }

        results = [None] * len(simulation.frames)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Performing density calculations:", unit="frame(s)"):
            i = futures[fut]
            results[i] = fut.result()

    results_chunks = get_n_even_chunks(results, averaging_blocks)

    elements = results[0]["elemental_number_density"].keys()

    density = []
    number_density = []
    average_density = []
    elemental_density = {element: [] for element in elements}

    for r_chunk in results_chunks:

        density.append(
            np.mean(np.stack([r["density"] for r in r_chunk]), axis=0)
        )

        number_density.append(
            np.mean(np.stack([r["number_density"] for r in r_chunk]), axis=0)
        )

        average_density.append(
            np.mean([r["average_density"] for r in r_chunk])
        )

        for element in elements:
            elemental_density[element].append(
                np.mean(
                    np.stack(
                        [r["elemental_number_density"][element] for r in r_chunk]
                    ),
                    axis=0
                )
            )

    density = np.stack(density)
    number_density = np.stack(number_density)
    average_density = np.array(average_density)

    for element in elements:
        elemental_density[element] = np.stack(elemental_density[element])

    return {
        "bin_edges": results[0]["bin_edges"],
        "bin_centers": results[0]["bin_centers"],
        "bin_volumes": results[0]["bin_volumes"],

        "density_mean": np.mean(density, axis=0),
        "density_std": np.std(density, axis=0, ddof=1),

        "number_density_mean": np.mean(number_density, axis=0),
        "number_density_std": np.std(number_density, axis=0, ddof=1),

        "average_density_mean": np.mean(average_density),
        "average_density_std": np.std(average_density, ddof=1),

        "elemental_number_density_mean": {
            element: np.mean(arr, axis=0)
            for element, arr in elemental_density.items()
        },

        "elemental_number_density_std": {
            element: np.std(arr, axis=0)
            for element, arr in elemental_density.items()
        }
    }