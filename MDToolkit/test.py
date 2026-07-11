import os 
from MDToolkit.data.objects import Topology, Frame, Simulation, LAMMPS_CustomDump_Reader
from MDToolkit.data.misc_objects import BoxVolume
from MDToolkit.utils.structure_file_utils import create_elements_dictionary
from MDToolkit.analysis.density import axial_density, axial_density_time_averaged
from MDToolkit.IO.read_file import lammps_data_file_to_topology, lammps_data_file_to_frame

filedir = "/media/jrjoseph/Elements/projects/training/graphene_water_kcl_ls6/"
filename = "Graphene_Water_KCl_2.5V_relaxed.data"

filepath = os.path.join(filedir, filename)

frame = lammps_data_file_to_frame(filepath)

print(frame.topology.charges)