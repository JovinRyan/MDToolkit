import numpy as np
import scipy.constants as sc
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from MDToolkit.data.objects import Simulation, StructuredSystem, Molecule, Atom
from MDToolkit.utils.misc_utils import sort_atom_list_by_index
from MDToolkit.data.misc_objects import Volume, BoxVolume, CylinderVolume

def radial_velocity_profile(system: StructuredSystem, cyl_volume : CylinderVolume, bins = 250):
    '''
    '''

    atoms_list = system.get_atoms_list()

    e_prop_keys = atoms_list[0].elemental_properties_keys

    if not any(k in e_prop_keys for k in ("vx", "vy", "vz")):
        raise ValueError(
            "At least one velocity component ('vx', 'vy', or 'vz') is required."
        )
    
    annuli = cyl_volume.discretize_radial(bins)

    radii = []
    mean_vx = []
    mean_vy = []
    mean_vz = []
    mean_speed = []

    for i, annulus in enumerate(annuli):

        if i == len(annuli) - 1:
            mask = annulus.contains_atoms(atoms_list, outer_radial_bound="closed")
        else:
            mask = annulus.contains_atoms(atoms_list)

        atoms = [atom for atom, keep in zip(atoms_list, mask) if keep]

        if not atoms:
            mean_vx.append(np.nan)
            mean_vy.append(np.nan)
            mean_vz.append(np.nan)
            mean_speed.append(np.nan)
            continue

        vx = np.array([a.elemental_properties.get("vx", 0.0) for a in atoms])
        vy = np.array([a.elemental_properties.get("vy", 0.0) for a in atoms])
        vz = np.array([a.elemental_properties.get("vz", 0.0) for a in atoms])

        speed = np.sqrt(vx**2 + vy**2 + vz**2)

        mean_vx.append(vx.mean())
        mean_vy.append(vy.mean())
        mean_vz.append(vz.mean())
        mean_speed.append(speed.mean())

    bin_edges = np.array([annuli[0].i_radius] + [a.o_radius for a in annuli])

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "vx": np.asarray(mean_vx),
        "vy": np.asarray(mean_vy),
        "vz": np.asarray(mean_vz),
        "speed": np.asarray(mean_speed),
    }

def radial_velocity_profile_time_averaged(simulation: Simulation, cyl_volume: CylinderVolume, bins = 250, n_workers = 32):
    '''
    '''

    with ProcessPoolExecutor(max_workers = n_workers) as executor:
        futures = {
            executor.submit(radial_velocity_profile, frame, cyl_volume, bins): i
            for i, frame in enumerate(simulation.frames)
        }

        results = [None] * len(simulation.frames)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Performing velocity profile calculations:", unit="frame(s)"):
            i = futures[fut]
            results[i] = fut.result()

    vx = np.stack([r["vx"] for r in results])
    vy = np.stack([r["vy"] for r in results])
    vz = np.stack([r["vz"] for r in results])
    speed = np.stack([r["speed"] for r in results])
    
    return {
        "bin_edges": results[0]["bin_edges"],
        "bin_centers": results[0]["bin_centers"],
        "vx": np.nanmean(vx, axis=0),
        "vx_std": np.nanstd(vx, axis=0),
        "vy": np.nanmean(vy, axis=0),
        "vy_std": np.nanstd(vy, axis=0),
        "vz": np.nanmean(vz, axis=0),
        "vz_std": np.nanstd(vz, axis=0),
        "speed": np.nanmean(speed, axis=0),
        "speed_std": np.nanstd(speed, axis=0),
    }