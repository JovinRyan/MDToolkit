import os 
from MDToolkit.data.objects import Topology, Frame, Simulation, LAMMPS_CustomDump_Reader
from MDToolkit.data.misc_objects import BoxVolume
from MDToolkit.utils.structure_file_utils import create_elements_dictionary
from MDToolkit.analysis.density import axial_density, axial_density_time_averaged
from MDToolkit.IO.read_file import lammps_data_file_to_topology, lammps_data_file_to_frame, packmol_pdb_file_to_frame, cif_file_to_frame
from MDToolkit.IO.write_file import write_lammps_data_file
from MDToolkit.paths import OUTPUT, CIF_FILES


filedir = CIF_FILES
filename = "MoS2.cif"

filepath = os.path.join(filedir, filename)

frame = packmol_pdb_file_to_frame(filepath)

print(frame.positions)