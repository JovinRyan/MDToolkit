import scipy.constants as sc
import numpy as np
from tqdm.auto import tqdm
from collections.abc import Sequence
from MDToolkit.data.objects import Frame
from MDToolkit.data.misc_objects import Volume
from MDToolkit.utils.misc_utils import get_n_even_chunks

def axial_density(frame : Frame, volumes : Sequence[Volume] = None, bin_width = 0.1, axis_idx = 0):
    '''
    '''

    if axis_idx > 2 or axis_idx < 0:
        raise ValueError("axis_idx must be <= 2 and >= 0")

    if volumes is None:
        volumes = [frame.box]

    masks = [vol.contains(frame.positions) for vol in volumes]

    mask = np.logical_or.reduce(masks)

    masked_axial_positions = frame.positions[mask][:, axis_idx]
    masked_masses = frame.get_masses()[mask]
    masked_types = frame.types[mask]

    axial_min = min(vol.bounding_box[0][axis_idx] for vol in volumes)
    axial_max = max(vol.bounding_box[1][axis_idx] for vol in volumes)

    edges = np.arange(axial_min, axial_max + bin_width, bin_width)

    counts, _ = np.histogram(
        masked_axial_positions,
        bins=edges
    )

    mass_per_bin, _ = np.histogram(
        masked_axial_positions,
        bins=edges,
        weights=masked_masses
    )

    unique_types = np.unique(masked_types)

    elemental_counts = {}

    for atom_type in unique_types:
        type_mask = masked_types == atom_type

        elemental_counts[atom_type], _ = np.histogram(
            masked_axial_positions[type_mask],
            bins=edges
        )

    bin_volumes = np.zeros(len(edges) - 1, dtype=np.float64)

    for i in range(len(bin_volumes)):

        bin_min = edges[i]
        bin_max = edges[i + 1]

        for vol in volumes:

            mins, maxs = vol.bounding_box

            vol_min = mins[axis_idx]
            vol_max = maxs[axis_idx]

            overlap = (
                min(bin_max, vol_max)
                - max(bin_min, vol_min)
            )

            if overlap <= 0:
                continue

            axial_length = vol_max - vol_min

            bin_volumes[i] += (
                overlap
                * vol.volume
                / axial_length
            )

    number_density = np.full(len(counts), np.nan, dtype=np.float64)
    mass_density = np.full(len(counts), np.nan, dtype=np.float64)

    valid = bin_volumes > 0

    number_density[valid] = (
        counts[valid]
        / bin_volumes[valid]
    )

    mass_density[valid] = (
        mass_per_bin[valid]
        * 1e24
        / (sc.N_A * bin_volumes[valid])
    )

    elemental_number_density = {}

    for atom_type, counts in elemental_counts.items():
        density = np.full_like(bin_volumes, np.nan, dtype=np.float64)
        density[valid] = counts[valid] / bin_volumes[valid]

        elemental_number_density[atom_type] = density

    centers = 0.5 * (edges[:-1] + edges[1:])

    return {
        "bin_edges": edges,
        "bin_centers": centers,
        "bin_volumes": bin_volumes,
        "number_density": number_density,
        "mass_density": mass_density,
        "elemental_number_density" : elemental_number_density
    }
    
def axial_density_time_averaged(simulation, volumes : Sequence[Volume] = None, bin_width = 0.1, axis_idx = 0, n_averaging_blocks = 10):
    '''
    '''

    results = []

    for frame in tqdm(simulation, total = len(simulation)):

        results.append(
            axial_density(
                frame,
                volumes = volumes,
                bin_width = bin_width,
                axis_idx = axis_idx
            )
        )
    
    r_chunks = get_n_even_chunks(results, n_averaging_blocks)

    number_density = np.stack([
        np.stack([r["number_density"] for r in chunk]).mean(axis=0)
        for chunk in r_chunks
    ])

    mass_density = np.stack([
        np.stack([r["mass_density"] for r in chunk]).mean(axis=0)
        for chunk in r_chunks
    ])

    atom_types = results[0]["elemental_number_density"].keys()

    elemental_number_density = {}

    for atom_type in atom_types:
        block_means = np.stack([
            np.stack([
                r["elemental_number_density"][atom_type]
                for r in chunk
            ]).mean(axis=0)
            for chunk in r_chunks
        ])

        elemental_number_density[atom_type] = {
            "mean": np.mean(block_means, axis=0),
            "std": np.std(block_means, axis=0, ddof=1)
        }


    return {
        "bin_edges": results[0]["bin_edges"],
        "bin_centers": results[0]["bin_centers"],
        "bin_volumes": results[0]["bin_volumes"],

        "number_density": {
            "mean": np.mean(number_density, axis = 0),
            "std": np.std(number_density, axis = 0, ddof = 1)
        },

        "mass_density": {
            "mean": np.mean(mass_density, axis = 0),
            "std": np.std(mass_density, axis = 0, ddof = 1)
        },

        "elemental_number_density": elemental_number_density
    }


# def radial_density(system : StructuredSystem, cyl_volume : CylinderVolume, bins = 250)-> dict:
#     '''
#     '''

#     annuli = cyl_volume.discretize_radial(bins)

#     system.populate_elemental_properties_for_all_atoms()

#     atoms_list = system.get_atoms_list()

#     atoms_in_annulus = []

#     for i in range(len(annuli)):
#         if i == len(annuli) - 1:
#             mask = annuli[i].contains_atoms(atoms_list, outer_radial_bound="closed")
#         else:
#             mask = annuli[i].contains_atoms(atoms_list)
        
#         atoms_in_annulus.append([atom for atom, keep in zip(atoms_list, mask) if keep])
    
#     atoms_per_bin = np.array([len(atoms)for atoms in atoms_in_annulus])

#     total_atoms = np.sum(atoms_per_bin)

#     bin_volumes = np.array([annulus.volume for annulus in annuli])

#     masses_in_bin = np.array([sum(atom.elemental_properties["AtomicMass"] for atom in atoms) for atoms in atoms_in_annulus])

#     density_in_bin = np.full(bins, 0.0)

#     mask = bin_volumes > 0

#     density_in_bin[mask] = (masses_in_bin[mask] * 1e24 / (sc.N_A * bin_volumes[mask]))

#     average_density = np.sum(density_in_bin[mask] * bin_volumes[mask]) / np.sum(bin_volumes[mask])

#     elements_in_bin = []

#     for atoms in atoms_in_annulus:

#         elements = [atom.element for atom in atoms]

#         elems, counts = np.unique(
#             elements,
#             return_counts=True
#         )

#         elements_in_bin.append(
#             dict(zip(elems, counts))
#         )

#     all_elements = sorted({
#         atom.element
#         for atom in atoms_list
#     })

#     elemental_number_density = {
#         e: np.zeros(bins)
#         for e in all_elements
#     }

#     for i, entry in enumerate(elements_in_bin):
#         for element, count in entry.items():
#             elemental_number_density[element][i] = count

#     for element in elemental_number_density:
#         elemental_number_density[element] = elemental_number_density[element] / bin_volumes
#         # total = elemental_number_density[element].sum()
#         # if total > 0:
#         #     elemental_number_density[element] /= total

#     bin_edges = np.array([annuli[0].i_radius] + [a.o_radius for a in annuli])

#     bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

#     number_density = atoms_per_bin / bin_volumes

#     return {
#         "bin_edges": bin_edges,
#         "bin_centers": bin_centers,
#         "bin_volumes": bin_volumes,
#         "number_density": number_density,
#         "elemental_number_density": {
#             k: v
#             for k, v in elemental_number_density.items()
#         },
#         "density": density_in_bin,
#         "average_density": average_density,
#     }

# def radial_density_time_averaged(simulation : Simulation, cyl_volume: CylinderVolume, bins = 250, n_workers = os.cpu_count()//2, averaging_blocks = 10) -> dict:
#     '''
#     '''

#     with ProcessPoolExecutor(max_workers = n_workers) as executor:
#         futures = {
#             executor.submit(radial_density, frame, cyl_volume, bins): i
#             for i, frame in enumerate(simulation.frames)
#         }

#         results = [None] * len(simulation.frames)

#         for fut in tqdm(as_completed(futures), total=len(futures), desc="Performing density calculations:", unit="frame(s)"):
#             i = futures[fut]
#             results[i] = fut.result()

#     results_chunks = get_n_even_chunks(results, averaging_blocks)

#     elements = results[0]["elemental_number_density"].keys()

#     density = []
#     number_density = []
#     average_density = []
#     elemental_density = {element: [] for element in elements}

#     for r_chunk in results_chunks:

#         density.append(
#             np.mean(np.stack([r["density"] for r in r_chunk]), axis=0)
#         )

#         number_density.append(
#             np.mean(np.stack([r["number_density"] for r in r_chunk]), axis=0)
#         )

#         average_density.append(
#             np.mean([r["average_density"] for r in r_chunk])
#         )

#         for element in elements:
#             elemental_density[element].append(
#                 np.mean(
#                     np.stack(
#                         [r["elemental_number_density"][element] for r in r_chunk]
#                     ),
#                     axis=0
#                 )
#             )

#     density = np.stack(density)
#     number_density = np.stack(number_density)
#     average_density = np.array(average_density)

#     for element in elements:
#         elemental_density[element] = np.stack(elemental_density[element])

#     return {
#         "bin_edges": results[0]["bin_edges"],
#         "bin_centers": results[0]["bin_centers"],
#         "bin_volumes": results[0]["bin_volumes"],

#         "density_mean": np.mean(density, axis=0),
#         "density_std": np.std(density, axis=0, ddof=1),

#         "number_density_mean": np.mean(number_density, axis=0),
#         "number_density_std": np.std(number_density, axis=0, ddof=1),

#         "average_density_mean": np.mean(average_density),
#         "average_density_std": np.std(average_density, ddof=1),

#         "elemental_number_density_mean": {
#             element: np.mean(arr, axis=0)
#             for element, arr in elemental_density.items()
#         },

#         "elemental_number_density_std": {
#             element: np.std(arr, axis=0)
#             for element, arr in elemental_density.items()
#         }
#     }