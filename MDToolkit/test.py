import os 
from MDToolkit.data.objects import Topology, Frame, Simulation, LAMMPS_CustomDump_Reader
from MDToolkit.data.misc_objects import BoxVolume
from MDToolkit.custom_systems.liquids import create_simplesalt_in_water_box
from MDToolkit.IO.read_file import lammps_data_file_to_topology, lammps_data_file_to_frame, packmol_pdb_file_to_frame, cif_file_to_frame
from MDToolkit.IO.write_file import write_lammps_data_file
from MDToolkit.paths import OUTPUT, CIF_FILES, PDB_FILES

water_box_bounds_1 = {
  "min_x" : -52.0, "max_x" : -2.0,
  "min_y" : -25.0, "max_y" : 25.0,
  "min_z" : -25.0, "max_z" : 25.0
}

aq_salt1 = create_simplesalt_in_water_box(water_box_bounds_1, "KCl", 1.0, solution_density=1.04, H2O_geometry="SPCE", density_correction=0.2)
aq_salt1.set_box_from_positions()
aq_salt1.set_molecule_bonds_by_type_indices()

print(aq_salt1.get_COM())