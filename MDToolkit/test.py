from MDToolkit.IO.read_file import packmol_pdb_file_to_frame
from MDToolkit.IO.write_file import write_lammps_data_file
from MDToolkit.utils.structure_file_utils import create_elements_dictionary
from MDToolkit.data.objects import Topology, Frame

file = "/oden/jrjoseph/Downloads/graphene_50_50.pdb"

type_mapping = {1 : "C"}

topol = Topology(type_mapping, create_elements_dictionary())

frame = packmol_pdb_file_to_frame(file, topol)

frame.set_box_from_positions()

# print(frame.positions)

write_lammps_data_file(frame, "test.data")