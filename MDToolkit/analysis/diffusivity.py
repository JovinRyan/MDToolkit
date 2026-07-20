import numpy as np
from tqdm.auto import tqdm
from MDToolkit.data.objects import Simulation, Frame, Topology
from MDToolkit.utils.misc_utils import get_n_even_chunks

# def compute_MSD(simulation : Simulation, averaging_bins = 10, subtract_COM = True):
#     '''
#     '''

#     timestep_chunks = get_n_even_chunks(simulation.timesteps, averaging_bins)
#     frames_chunks = get_n_even_chunks(simulation.frames, averaging_bins)
#     atom_counts_chunks = get_n_even_chunks(simulation.atom_counts, averaging_bins)

#     all_msds = []
#     for i in range(len(frames_chunks)):
#         reference_frame = frames_chunks[i][0]

#         reference_frame_coordinates_matrix = np.vstack([
#             np.array(atom.position)
#             for molecule in reference_frame.molecule_list
#             for atom in molecule.atoms
#         ])

#         if subtract_COM:
#             reference_COM = np.array(reference_frame.find_COM())

#         msds = []

#         for j in range(len(frames_chunks[i])):
#             frame = frames_chunks[i][j]

#             coordinates_matrix = np.vstack([
#                 np.array(atom.position)
#                 for molecule in frame.molecule_list
#                 for atom in molecule.atoms
#             ])

#             if subtract_COM:
#                 r_t_corr = coordinates_matrix - np.array(frame.find_COM())
#                 r_0_corr = reference_frame_coordinates_matrix - reference_COM

#                 dp_matrix = (r_t_corr - r_0_corr) ** 2
#             else:
#                 dp_matrix = (coordinates_matrix - reference_frame_coordinates_matrix) ** 2

#             msds.append(np.sum(dp_matrix, axis=0) / atom_counts_chunks[i][j])  # per atom scalar MSD

#             msds[j] = np.append(msds[j], np.sum(msds[j]))

#         all_msds.append(msds)

#     mean_msds = np.mean(all_msds, axis = 0)

#     msds_std_dev = np.std(all_msds, axis = 0)


#     return {
#         "x_msd" : mean_msds[:, 0], "x_std" : msds_std_dev[:, 0],
#         "y_msd" : mean_msds[:, 1], "y_std" : msds_std_dev[:, 1],
#         "z_msd" : mean_msds[:, 2], "z_std" : msds_std_dev[:, 2],
#         "sum_msd" : mean_msds[:, 3], "sum_std" : msds_std_dev[:, 3],
#         "timesteps" : timestep_chunks[0]
#     }

def compute_msd(simulation: Simulation, subtract_COM = True, n_averaging_blocks = 10):
    '''
    '''
    indices = np.arange(len(simulation))

    index_chunks = get_n_even_chunks(indices, n_averaging_blocks)

    timesteps = get_n_even_chunks(
        np.array([frame.timestep for frame in simulation]),
        n_averaging_blocks
    )[0]

    msd_chunks = []

    used_unwrapped_positions = True

    for index_chunk in tqdm(index_chunks):

        reference_frame = simulation[index_chunk[0]]

        if subtract_COM:
            reference_COM = reference_frame.get_COM()
        else:
            reference_COM = np.array([0.0, 0.0, 0.0])

        block_msd = []

        for i in index_chunk:

            frame = simulation[i]

            if subtract_COM:
                COM = frame.get_COM()
            else:
                COM = np.array([0.0, 0.0, 0.0])

            if frame.unwrapped_positions is not None:
                diff = frame.unwrapped_positions - reference_frame.unwrapped_positions
            else:
                used_unwrapped_positions = False
                diff = frame.positions - reference_frame.positions

            block_msd.append(
                np.mean((diff - COM + reference_COM) ** 2, axis = 0)
            )

        block_msd = np.array(block_msd)

        block_msd_sum = block_msd.sum(axis = 1, keepdims = True)

        block_msd = np.hstack((block_msd, block_msd_sum))

        msd_chunks.append(block_msd)

    if not used_unwrapped_positions:
        print("Warning: used wrapped positions for MSD calculation!")

    msd_chunks = np.array(msd_chunks)

    msd_mean = np.mean(msd_chunks, axis = 0)
    msd_std = np.std(msd_chunks, axis = 0)

    return {
        "msd": msd_mean,
        "std": msd_std,
        "t": timesteps
    }

    

    

