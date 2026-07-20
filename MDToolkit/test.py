import os 
from MDToolkit.data.objects import Topology, Frame, Simulation, LAMMPS_CustomDump_Reader, MultiSimulation
from MDToolkit.utils.structure_file_utils import create_elements_dictionary
from MDToolkit.data.misc_objects import BoxVolume
from MDToolkit.custom_systems.liquids import create_simplesalt_in_water_box
from MDToolkit.IO.read_file import lammps_data_file_to_topology, lammps_data_file_to_frame, packmol_pdb_file_to_frame, cif_file_to_frame
from MDToolkit.IO.write_file import write_lammps_data_file
from MDToolkit.paths import OUTPUT, CIF_FILES, PDB_FILES

filedir = "/media/jrjoseph/Elements/projects/kcl_mos2_ls6/"

filenames = ["kcl_mos2_equilibration_run0.5.out", "kcl_mos2_equilibration_run0.75.out", "kcl_mos2_equilibration_run1.out"]

filepaths = [os.path.join(filedir, name) for name in filenames]

type_mapping = {
    1 : "Mo",
    2 : "S",
    3 : "Cl",
    4 : "H",
    5 : "K",
    6 : "O"
}

topol = Topology(type_mapping, elements_dict = create_elements_dictionary())

simulation = MultiSimulation(filepaths, topol, LAMMPS_CustomDump_Reader)