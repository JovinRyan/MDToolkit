from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

from MDToolkit.data.misc_objects import Volume
from MDToolkit.data.objects import Simulation, StructuredSystem


def get_frame_ion_v_membership(system: StructuredSystem, ions_list, v1: Volume, v_transition: Volume, v2: Volume):
    '''
    '''

    v1_mask = v1.contains_atoms(ions_list)
    vt_mask = v_transition.contains_atoms(ions_list)
    v2_mask = v2.contains_atoms(ions_list)

    membership = np.column_stack((v1_mask, vt_mask, v2_mask))

    return membership


def translocation_current(simulation: Simulation, ion_spcs: list[str], v1: Volume, v_transition: Volume, v2: Volume):
    '''
    '''

    simulation.timesteps = simulation.timesteps - simulation.timesteps[0]

    atoms_list = simulation.frames[0].get_atoms_list()

    ions_list = []

    for atom in atoms_list:
        if atom.element in ion_spcs:
            ions_list.append(atom)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(get_frame_ion_v_membership, frame, ions_list, v1, v_transition, v2): i
            for i, frame in enumerate(simulation.frames)
        }

        results = [None] * len(simulation.frames)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Finding ion positions:", unit="frame(s)"):
            i = futures[fut]
            results[i] = fut.result()

    membership = np.stack(results)

    in_v1 = membership[:, :, 0]
    in_vt = membership[:, :, 1]
    in_v2 = membership[:, :, 2]

    n_frames, n_ions = in_v1.shape

    state = np.zeros(n_ions, dtype=np.int8)

    # 0 = idle
    # 1 = started in V1
    # 2 = V1 -> VT observed
    # 3 = started in V2
    # 4 = V2 -> VT observed

    events = []

    delta_q = np.zeros(n_frames)

    for f in range(n_frames):

        # New ions beginning in V1
        state[(state == 0) & in_v1[f]] = 1

        # New ions beginning in V2
        state[(state == 0) & in_v2[f]] = 3

        # Enter transition region
        state[(state == 1) & in_vt[f]] = 2
        state[(state == 3) & in_vt[f]] = 4

        # Successful crossings
        forward = (state == 2) & in_v2[f]
        backward = (state == 4) & in_v1[f]

        for ion_idx in np.where(forward)[0]:

            events.append({
                "frame": f,
                "time": simulation.timesteps[f],
                "ion_idx": ion_idx,
                "species": ions_list[ion_idx].element,
                "charge": ions_list[ion_idx].charge,
                "direction": "V1->V2"
            })

            delta_q[f] += ions_list[ion_idx].charge

        for ion_idx in np.where(backward)[0]:

            events.append({
                "frame": f,
                "time": simulation.timesteps[f],
                "ion_idx": ion_idx,
                "species": ions_list[ion_idx].element,
                "charge": ions_list[ion_idx].charge,
                "direction": "V2->V1"
            })

            delta_q[f] -= ions_list[ion_idx].charge

        # Reset completed ions
        state[forward | backward] = 0

        # Returned to original side
        state[(state == 2) & in_v1[f]] = 1
        state[(state == 4) & in_v2[f]] = 3

    cumulative_q = np.cumsum(delta_q)

    return events, delta_q, cumulative_q

def charge_velocity_current(simulation, averaging_blocks = 10, current_vector = [1, 0, 0]):
    '''
    '''
    box_dims_dict = simulation.frames[0].box_dimensions
    
    if current_vector == [1, 0, 0]:
        L = box_dims_dict["max_x"] - box_dims_dict["min_x"]
        v_key = "vx"
    elif current_vector == [0, 1, 0]:
        L = box_dims_dict["max_y"] - box_dims_dict["min_y"]
        v_key = "vy"
    else:
        L = box_dims_dict["max_z"] - box_dims_dict["min_z"]
        v_key = "vz"

    atoms_lists = [frame.get_atoms_list() for frame in simulation.frames]

    I = []
    for atoms_list in tqdm(atoms_lists):
        v = np.empty(len(atoms_list))
        q = np.empty(len(atoms_list))

        for i, atom in enumerate(atoms_list):
            v[i] = atom.elemental_properties[v_key]
            q[i] = atom.charge
        
        I.append(np.sum(q*v / L))

    return I