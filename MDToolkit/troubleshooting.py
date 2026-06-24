import cProfile
from MDToolkit.IO.read_file import lammps_dump_file_to_simulation
# Importing System

LAMMPS_dump_file_path = "/media/jrjoseph/Elements/projects/training/water_box_ls6/water_box_nvt_prod_coarse.out"

type_mapping = {
    1 : "O",
    2 : "H"
}

water_box_simulation = lammps_dump_file_to_simulation(LAMMPS_dump_file_path, type_mapping)
