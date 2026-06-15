import numpy as np
from MDToolkit.data.objects import Simulation
from MDToolkit.utils.misc_utils import get_n_even_chunks

def compute_MSD(simulation : Simulation, averaging_bins = 10, subtract_COM = True):
    '''
    '''

    timestep_chunks = get_n_even_chunks(simulation.timesteps, averaging_bins)
    frames_chunks = get_n_even_chunks(simulation.frames, averaging_bins)
    atom_counts_chunks = get_n_even_chunks(simulation.atom_counts, averaging_bins)

    all_msds = []
    for i in range(len(frames_chunks)):
        reference_frame = frames_chunks[i][0]

        reference_frame_coordinates_matrix = np.vstack([
            np.array(atom.position)
            for molecule in reference_frame.molecule_list
            for atom in molecule.atoms
        ])

        if subtract_COM:
            reference_COM = np.array(reference_frame.find_COM())

        msds = []

        for j in range(len(frames_chunks[i])):
            frame = frames_chunks[i][j]

            coordinates_matrix = np.vstack([
                np.array(atom.position)
                for molecule in frame.molecule_list
                for atom in molecule.atoms
            ])

            if subtract_COM:
                r_t_corr = coordinates_matrix - np.array(frame.find_COM())
                r_0_corr = reference_frame_coordinates_matrix - reference_COM

                dp_matrix = (r_t_corr - r_0_corr) ** 2
            else:
                dp_matrix = (coordinates_matrix - reference_frame_coordinates_matrix) ** 2

            msds.append(np.sum(dp_matrix, axis=0) / atom_counts_chunks[i][j])  # per atom scalar MSD

            msds[j] = np.append(msds[j], np.sum(msds[j]))

        all_msds.append(msds)

    mean_msds = np.mean(all_msds, axis = 0)

    msds_std_dev = np.std(all_msds, axis = 0)


    return {
        "x_msd" : mean_msds[:, 0], "x_std" : msds_std_dev[:, 0],
        "y_msd" : mean_msds[:, 1], "y_std" : msds_std_dev[:, 1],
        "z_msd" : mean_msds[:, 2], "z_std" : msds_std_dev[:, 2],
        "sum_msd" : mean_msds[:, 3], "sum_std" : msds_std_dev[:, 3],
        "timesteps" : timestep_chunks[0]
    }
    