import cProfile
from MDToolkit.IO.read_file import lammps_dump_file_to_simulation
# Importing System

LAMMPS_dump_file_path = "/media/jrjoseph/Elements/projects/training/cnt_water_ls6/density_calculation/cnt_water_nvt_prod.out"

type_mapping = {
    1 : "H",
    2 : "C",
    3 : "O"
}

water_box_simulation = lammps_dump_file_to_simulation(LAMMPS_dump_file_path, type_mapping)

idxs = [1, 45, 230, 444]


for i in range(3):
    atoms_list = water_box_simulation.frames[i].get_atoms_list()
    for idx in idxs:
        print(atoms_list[idx].id)