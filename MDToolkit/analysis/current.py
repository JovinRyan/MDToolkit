import numpy as np
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor
from MDToolkit.data.objects import Frame, Simulation
from MDToolkit.utils.misc_utils import get_n_even_chunks
from MDToolkit.data.misc_objects import Volume

def qv_current(frame : Frame, ions_spcs : list[str], I_vector = [1, 0, 0]):
    '''
    '''
    vec_idx = I_vector.index(1)

    ions_types = [k for k, v in frame.topology.type_mapping.items() if v in ions_spcs]

    mask = np.isin(frame.types, ions_types)

    if vec_idx > 2 or vec_idx < 0:
        raise ValueError("I_vector must be [1, 0, 0], [0, 1, 0], or [0, 0, 1]")

    qs = frame.get_charges()[mask]

    vs = frame.velocities[:, vec_idx][mask]

    L = frame.box.dims[vec_idx]

    return np.sum(qs*vs) / L

def qv_current_time_averaged(simulation : Simulation, ion_spcs : list[str], I_vector = [1, 0, 0], n_averaging_blocks = 10):
    '''
    '''
    vec_idx = I_vector.index(1)

    if vec_idx > 2 or vec_idx < 0:
        raise ValueError("I_vector must be [1, 0, 0], [0, 1, 0], or [0, 0, 1]")
    
    I = []
    
    for frame in tqdm(simulation, total = len(simulation)):
        I.append(qv_current(frame, ion_spcs, I_vector))
    
    I = get_n_even_chunks(I, n_averaging_blocks)

    I = [np.mean(chunk) for chunk in I]

    return {
        "I" : np.mean(I),
        "std" : np.std(I)
    }

    
def frame_ion_volume_membership(frame : Frame, ion_spcs : list[str], v1 : Volume, v_transition : Volume, v2 : Volume):
    '''
    '''

    ions_types = [k for k, v in frame.topology.type_mapping.items() if v in ion_spcs]

    ion_mask = np.isin(frame.types, ions_types)

    membership = np.column_stack((np.logical_and(ion_mask, v1.contains(frame.positions)), np.logical_and(ion_mask, v_transition.contains(frame.positions)), np.logical_and(ion_mask, v2.contains(frame.positions))))

    return membership

def translocation_current(simulation : Simulation, ion_spcs : list[str], v1 : Volume, v_transition : Volume, v2 : Volume):
    '''
    '''

    membership = []
    timesteps = []

    for frame in tqdm(simulation):

        membership.append(
            frame_ion_volume_membership(
                frame,
                ion_spcs,
                v1,
                v_transition,
                v2
            )
        )

        timesteps.append(frame.timestep)

    membership = np.stack(membership)

    first_frame = simulation[0]

    ion_types = [
        t
        for t, s in first_frame.topology.type_mapping.items()
        if s in ion_spcs
    ]

    ion_mask = np.isin(first_frame.types, ion_types)

    charges = first_frame.get_charges()[ion_mask]

    species = np.array([
        first_frame.topology.type_mapping[t]
        for t in first_frame.types[ion_mask]
    ])

    membership = membership[:, ion_mask, :]

    in_v1 = membership[:, :, 0]
    in_vt = membership[:, :, 1]
    in_v2 = membership[:, :, 2]

    n_frames, n_ions = in_v1.shape

    state = np.zeros(n_ions, dtype=np.int8)

    charge_translocation = np.zeros(n_frames)

    ion_translocations = {
        spc: np.zeros(n_frames, dtype=np.int32)
        for spc in ion_spcs
    }

    # 0 = idle
    # 1 = started in V1
    # 2 = V1 -> VT observed
    # 3 = started in V2
    # 4 = V2 -> VT observed

    for f in range(n_frames):

        state[(state == 0) & in_v1[f]] = 1
        state[(state == 0) & in_v2[f]] = 3

        state[(state == 1) & in_vt[f]] = 2
        state[(state == 3) & in_vt[f]] = 4

        forward = (state == 2) & in_v2[f]
        backward = (state == 4) & in_v1[f]

        charge_translocation[f] = (
            np.sum(charges[forward])
            - np.sum(charges[backward])
        )

        for spc in ion_spcs:

            spc_mask = species == spc

            ion_translocations[spc][f] = (
                np.count_nonzero(forward & spc_mask)
                - np.count_nonzero(backward & spc_mask)
            )

        state[forward | backward] = 0

        state[(state == 2) & in_v1[f]] = 1
        state[(state == 4) & in_v2[f]] = 3

    return {
        "timesteps": np.asarray(timesteps),
        "ion_translocations": ion_translocations,
        "charge_translocation": charge_translocation,
        "cumulative_charge_translocation": np.cumsum(charge_translocation)
    }