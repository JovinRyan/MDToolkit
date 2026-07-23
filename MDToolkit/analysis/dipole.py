import numpy as np 
from tqdm.auto import tqdm
from MDToolkit.data.objects import Frame, Simulation
from MDToolkit.data.misc_objects import Volume
from MDToolkit.utils.misc_utils import get_n_even_chunks

def frame_dipole_moment(frame : Frame, use_unwrapped_positions = True):
    '''
    '''
    warning_printed = False

    if use_unwrapped_positions and frame.unwrapped_positions is not None:
        pos = frame.unwrapped_positions
    else:
        pos = frame.positions
        if not warning_printed: 
            print("Warning: using wrapped coordinates for analysis")
            warning_printed = True
    
    charges = np.array(frame.get_charges())

    return charges @ pos

def system_dipole_moment(simulation: Simulation, use_unwrapped_positions = True):
    '''
    '''
    dipole_moment_vectors = []
    t = []
    for frame in tqdm(simulation):
        dipole_moment_vectors.append(frame_dipole_moment(frame, use_unwrapped_positions))
        t.append(frame.timestep)

    return {
        "dipole_moment_vectors" : dipole_moment_vectors,
        "t" : t
    }

def dipole_distribution(frame : Frame, reference_vector = [1, 0, 0], vol : Volume = None, abs = True):
    '''
    '''
    if vol is None:
        vol_mask = np.ones(len(frame.positions), dtype = bool)
    else:
        vol_mask = vol.contains(frame.positions)
    
    charges = np.array(frame.get_charges())
    masses = np.array(frame.get_masses())

    dipole_vectors = []
    coms = []

    for _, idxs in frame.iter_molecules(mask = vol_mask, mode = "whole"):
        q = charges[idxs]
        if np.isclose(np.sum(q), 0.0, atol = 1e-6):
            dipole_vectors.append(q @ frame.positions[idxs])
            coms.append(masses[idxs] @ frame.positions[idxs] / np.sum(masses[idxs]))

    reference_vector = np.asarray(reference_vector, dtype=float)
    reference_vector /= np.linalg.norm(reference_vector)

    dipole_vectors = np.asarray(dipole_vectors)

    dipole_magnitudes = np.linalg.norm(dipole_vectors, axis = 1)

    valid = dipole_magnitudes > 0

    angles = np.full(len(dipole_vectors), np.nan)

    cos_theta = (dipole_vectors[valid] @ reference_vector / dipole_magnitudes[valid])

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    if abs:
        angles[valid] = np.arccos(np.abs(cos_theta))
    else:
        angles[valid] = np.arccos(cos_theta)

    coms = np.asarray(coms)

    return {
        "angles" : angles[valid],
        "coms" : coms[valid]
    }

def dipole_angle_distribution_timeaveraged(simulation : Simulation, reference_vector = [1, 0, 0], vol : Volume = None, abs = True, n_bins = 100, n_averaging_blocks = 10):
    '''
    '''
    results = []

    for frame in tqdm(simulation):
        results.append(
            dipole_distribution(
                frame,
                reference_vector = reference_vector,
                vol = vol,
                abs = abs
            )
        )

    blocks = get_n_even_chunks(results, n_chunks = n_averaging_blocks)

    if abs:
        angle_range = (0, np.pi / 2)
    else:
        angle_range = (0, np.pi)

    bin_edges = np.linspace(
        angle_range[0],
        angle_range[1],
        n_bins + 1
    )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    block_distributions = []

    for block in blocks:

        angles = np.concatenate(
            [
                result["angles"]
                for result in block
            ]
        )

        distribution, _ = np.histogram(
            angles,
            bins = bin_edges,
            density = True
        )

        block_distributions.append(distribution)

    block_distributions = np.asarray(block_distributions)

    return {
        "angles" : bin_centers,
        "probability" : np.mean(block_distributions, axis = 0),
        "std" : np.std(block_distributions, axis = 0)
    }