from MDToolkit.data.misc_objects import CylinderVolume, BoxVolume
from MDToolkit.IO.read_file import lammps_dump_file_to_simulation, lammps_data_file_to_structured_system
from MDToolkit.analysis.density import radial_density_time_averaged
from MDToolkit.analysis.current import get_frame_ion_v_membership
import matplotlib.pyplot as plt

cyl = CylinderVolume([-25.174, 0, 0], [25.174, 0, 0], 8.0)

volumes = [BoxVolume([-104, -27.69, -27.15], [-53, 27.69, 27.15]), BoxVolume([53, -27.69, -27.15], [104, 27.69, 27.15])]

system = lammps_data_file_to_structured_system("/media/jrjoseph/Elements/projects/training/cnt_water_ls6/density_calculation/CNT_Water_final.data")

atoms_list = system.get_atoms_list()

o_list = []

for atom in atoms_list:
    if atom.element == "O":
        o_list.append(atom)


print(get_frame_ion_v_membership(system, o_list, volumes[0], cyl, volumes[1]))