import pandas as pd
import os
import copy
from MDToolkit.custom_systems.liquids import create_water_box, create_ionic_liquid_box
from MDToolkit.custom_systems.membranes import cif_file_to_monolayer_membrane
from MDToolkit.utils.cutom_systems_utils import create_pore_circular
from MDToolkit.IO.write_file import write_lammps_structure_file_atomic_full, write_pdb_file_from_StructuredSystem
from MDToolkit.IO.read_file import pdb_file_to_structured_system
from MDToolkit.paths import CIF_FILES, PDB_FILES

water_box_bounds = {
  "min_x" : 0.0, "max_x" : 30.0,
  "min_y" : 0.0, "max_y" : 30.0,
  "min_z" : 0.0, "max_z" : 30.0
}

charge_dict = {
    "H" : 0.6791, # TIP4P
    "O" : -1.3582 # TIP4P
}

water_box_system = create_water_box(water_box_bounds)

water_box_system.populate_elemental_properties_for_all_atoms()
water_box_system.populate_atom_charges(charge_dict)

write_lammps_structure_file_atomic_full(water_box_system, "water_box_3nmby3nmby3nm.data")
write_pdb_file_from_StructuredSystem(water_box_system, "water_box_3nmby3nmby3nm.pdb")