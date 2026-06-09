from MDToolkit.IO.read_file import pdb_file_to_structured_system
from MDToolkit.IO.write_file import write_lammps_structure_file_atomic_full

cnt_pdb_file = "/media/jrjoseph/Elements/projects/training/water_graphene_cnt/cnt.pdb"

cnt_system = pdb_file_to_structured_system(cnt_pdb_file)

write_lammps_structure_file_atomic_full(cnt_system, "cnt_lmp.data")