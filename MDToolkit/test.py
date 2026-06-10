from MDToolkit.IO.read_file import pdb_file_to_structured_system, lammps_data_file_to_structured_system
from MDToolkit.IO.write_file import write_lammps_structure_file_atomic_full

cnt_pdb_file = "/media/jrjoseph/Elements/projects/training/water_graphene_cnt/cnt.pdb"

lammps_water_box_data_file = "/media/jrjoseph/Elements/projects/training/water_box/equilibrated_water_box_spce.out"

print(lammps_data_file_to_structured_system(lammps_water_box_data_file))