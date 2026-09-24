import numpy as np
import os
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from MDToolkit.data.objects import Simulation, Frame, Topology
from MDToolkit.utils.misc_utils import get_n_even_chunks

def frame_msd(frame : Frame, reference_positions, ion_spcs : list[str], reference_COM = None):
    '''
    '''
    msd = {"timestep" : frame.timestep}

    if reference_COM is not None:
        COM_displacement = frame.get_COM() - reference_COM

    for ion_spc in ion_spcs:

        ion_types = [
            k for k, v in frame.topology.type_mapping.items()
            if v == ion_spc
        ]

        mask = np.isin(frame.types, ion_types)

        if frame.unwrapped_positions is not None:
            displacements = frame.unwrapped_positions[mask] - reference_positions[mask]
        else:
            displacements = frame.positions[mask] - reference_positions[mask]

        if reference_COM is not None:
            displacements -= COM_displacement

        msd[ion_spc] = np.mean(
            np.sum(np.square(displacements), axis = 1),
            axis = 0
        )

    return msd

_readers = None
_reference_positions = None
_reference_COM = None

def _initialize_msd_worker(metadata_list, topology, reference_positions, reference_COM):

    global _readers
    global _reference_positions
    global _reference_COM

    _readers = {}
    _reference_positions = reference_positions
    _reference_COM = reference_COM

    for metadata in metadata_list:

        reader = metadata["reader"](
            metadata["filepath"],
            topology,
            frame_offsets = metadata["frame_offsets"],
            filesize = metadata["filesize"]
        )

        _readers[metadata["filepath"]] = reader

def _frame_msd_worker(args):
    '''
    '''
    metadata, idx, ion_spcs = args

    reader = _readers[metadata["filepath"]]
    
    frame = reader.read_frame(idx)

    return frame_msd(frame, reference_positions=_reference_positions, ion_spcs=ion_spcs, reference_COM=_reference_COM)

def compute_msd(simulation: Simulation, ion_spcs : list[str], subtract_COM = True, n_workers = os.cpu_count() // 2):
    '''
    '''
    metadata = simulation.metadata
    
    if not isinstance(metadata, list):
        metadata = [metadata]

    reference_frame = simulation[0]

    ions_types = [
        k for k, v in reference_frame.topology.type_mapping.items()
        if v in ion_spcs
    ]

    reference_positions = reference_frame.unwrapped_positions

    reference_COM = reference_frame.get_COM() if subtract_COM else None

    tasks = [
        (frame_metadata, idx, ion_spcs)
        for frame_metadata, idx in simulation.iter_frame_tasks()
    ]

    with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_initialize_msd_worker,
            initargs=(metadata, simulation.topology, reference_positions, reference_COM)
        ) as executor:
    
            results = list(
                tqdm(
                    executor.map(_frame_msd_worker, tasks, chunksize=500),
                    total=len(tasks)
                )
            )
    results = {
        "timesteps": np.array([result["timestep"] for result in results]),
        **{
            ion_spc: np.array([
                result[ion_spc]
                for result in results
            ])
            for ion_spc in ion_spcs
        }
    }

    return results

def compute_diffusivity(msd_data, ion_spcs, n_blocks = 10):
    '''
    '''
    indices = get_n_even_chunks(
        range(len(msd_data["timesteps"])),
        n_chunks = n_blocks
    )

    diffusivities = {}

    timesteps = msd_data["timesteps"]

    for ion_spc in ion_spcs:

        block_diffusivities = []

        for block in indices:

            block = np.array(block)

            t = timesteps[block]
            t = t - t[0]

            msd = msd_data[ion_spc][block]
            msd = msd - msd[0]

            slope, intercept = np.polyfit(t, msd, 1)

            block_diffusivities.append(slope / 6)

        block_diffusivities = np.array(block_diffusivities)

        diffusivities[ion_spc] = {
            "blocks": block_diffusivities,
            "mean": np.mean(block_diffusivities),
            "std": np.std(block_diffusivities, ddof = 1)
        }

    return diffusivities