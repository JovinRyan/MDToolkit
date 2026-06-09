import os
from MDToolkit.IO.read_file import pdb_file_to_structured_system
from MDToolkit.IO.write_file import write_lammps_structure_file_atomic_full
from MDToolkit.utils.structure_file_utils import read_molecular_data_json_entry, get_alias_key_from_file_path
from MDToolkit.paths import PDB_FILES
from MDToolkit.custom_systems.liquids import create_water_box

water_box_bounds = {
  "min_x" : 0.0, "max_x" : 30.0,
  "min_y" : 0.0, "max_y" : 30.0,
  "min_z" : 0.0, "max_z" : 30.0
}

water_box_system = create_water_box(water_box_bounds, H2O_geometry="SPCE")

H2O_pdb_path = os.path.join(PDB_FILES, "H2O.pdb")

key = get_alias_key_from_file_path(H2O_pdb_path)

for molecule in water_box_system.molecule_list:
  molecule.populate_bonds_from_molecular_data(read_molecular_data_json_entry(key))

for molecule in water_box_system.molecule_list:
  molecule.populate_angles_from_molecular_data(read_molecular_data_json_entry(key))

water_box_system.reset_angle_ids()
water_box_system.reset_bond_ids()

write_lammps_structure_file_atomic_full(water_box_system, "water_box_3nmby3nmby3nm_SPCE.data")
