from MDToolkit.data.misc_objects import CylinderVolume
from MDToolkit.IO.read_file import lammps_dump_file_to_simulation
from MDToolkit.analysis.density import radial_density_time_averaged
import matplotlib.pyplot as plt

cyl = CylinderVolume([-25.174, 0, 0], [25.174, 0, 0], 8.0)

file_path = "/media/jrjoseph/Elements/projects/training/cnt_water_ls6/density_calculation/cnt_water_nvt_prod.out"

type_dict = {
    1 : "H",
    2 : "C",
    3 : "O"
}

simulation = lammps_dump_file_to_simulation(file_path, type_dict)
