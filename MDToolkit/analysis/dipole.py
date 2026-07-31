import numpy as np 
import os 
from concurrent.futures import ProcessPoolExecutor
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

def dipole_angle_distribution(frame: Frame, plane="xy", ref_vector = [1, 0, 0], vol: Volume = None):
    '''
    '''

    ref_vector = np.asarray(ref_vector, dtype = float)
    ref_vector /= np.linalg.norm(ref_vector)

    if vol is None:
        vol_mask = np.ones(len(frame.positions), dtype=bool)
    else:
        vol_mask = vol.contains(frame.positions)

    charges = np.asarray(frame.get_charges())
    masses = np.asarray(frame.get_masses())

    dipole_vectors = []
    coms = []

    for _, idxs in frame.iter_molecules(mask=vol_mask, mode="whole"):

        q = charges[idxs]

        if np.isclose(np.sum(q), 0.0, atol=1e-6):

            dipole_vectors.append(q @ frame.positions[idxs])

            coms.append(
                masses[idxs] @ frame.positions[idxs] / np.sum(masses[idxs])
            )

    dipole_vectors = np.asarray(dipole_vectors)
    coms = np.asarray(coms)

    if len(dipole_vectors) == 0:
        return {
            "angles": np.array([]),
            "coms": np.empty((0, 3))
        }

    plane = plane.lower()

    if plane == "xy":
        dipole_vectors[:, 2] = 0.0

    elif plane == "xz":
        dipole_vectors[:, 1] = 0.0

    elif plane == "yz":
        dipole_vectors[:, 0] = 0.0

    else:
        raise ValueError(
            "plane must be one of 'xy', 'xz', or 'yz'"
        )
    
    dipole_magnitudes = np.linalg.norm(dipole_vectors, axis = 1)
    valid = dipole_magnitudes > 0

    cos_theta = (dipole_vectors[valid] @ ref_vector) / dipole_magnitudes[valid]

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angles = np.degrees(np.arccos(cos_theta))

    return {
        "angles": angles,
        "coms": coms[valid]
    }

_readers = None

def _initialize_dipole_worker(metadata_list, topology):

    global _readers

    _readers = {}

    for metadata in metadata_list:

        reader = metadata["reader"](
            metadata["filepath"],
            topology,
            frame_offsets = metadata["frame_offsets"],
            filesize = metadata["filesize"]
        )

        _readers[metadata["filepath"]] = reader


def _dipole_angle_distribution_worker(args):

    metadata, idx, plane, ref_vector, vol = args

    reader = _readers[metadata["filepath"]]

    frame = reader.read_frame(idx)

    return dipole_angle_distribution(
        frame,
        plane = plane,
        ref_vector = ref_vector,
        vol = vol,
    )

def dipole_angle_distribution_timeaveraged(simulation : Simulation, plane = "xy", ref_vector = [1, 0, 0], vol : Volume = None, n_bins = 100, n_averaging_blocks = 10, n_workers = os.cpu_count() // 2):
    '''
    '''
    metadata = simulation.metadata

    if not isinstance(metadata, list):
        metadata = [metadata]

    tasks = [
        (frame_metadata, idx, plane, ref_vector, vol)
        for frame_metadata, idx in simulation.iter_frame_tasks()
    ]

    with ProcessPoolExecutor(
        max_workers = n_workers,
        initializer = _initialize_dipole_worker,
        initargs = (metadata, simulation.topology)
    ) as executor:

        results = list(
            tqdm(
                executor.map(
                    _dipole_angle_distribution_worker,
                    tasks
                ),
                total = len(tasks)
            )
        )

    blocks = get_n_even_chunks(
        results,
        n_chunks = n_averaging_blocks
    )

    bin_edges = np.linspace(
        0,
        180,
        n_bins + 1
    )

    bin_centers = 0.5 * (
        bin_edges[:-1] + bin_edges[1:]
    )

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

        block_distributions.append(
            distribution
        )

    block_distributions = np.asarray(
        block_distributions
    )

    mean = np.mean(
        block_distributions,
        axis = 0
    )

    std = np.std(
        block_distributions,
        axis = 0,
        ddof = 1
    )

    sem = std / np.sqrt(
        len(block_distributions)
    )

    return {
        "angles" : bin_centers,
        "probability" : mean,
        "std" : std,
        "sem" : sem
    }