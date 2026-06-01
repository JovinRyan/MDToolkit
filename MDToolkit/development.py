import pandas as pd
import os
import copy
from MDToolkit.custom_systems.liquids import create_water_box, create_ionic_liquid_box
from MDToolkit.custom_systems.membranes import cif_file_to_monolayer_membrane
from MDToolkit.utils.cutom_systems_utils import create_pore_circular
from MDToolkit.IO.write_file import write_lammps_structure_file_atomic_full, write_pdb_file_from_StructuredSystem
from MDToolkit.IO.read_file import pdb_file_to_structured_system
from MDToolkit.paths import CIF_FILES, PDB_FILES

C_cif_file_path = os.path.join(CIF_FILES, "graphene.cif")
OMIM_pdb_file_path = os.path.join(PDB_FILES, "OMIM.pdb")
PF6_pdb_file_path = os.path.join(PDB_FILES, "PF6.pdb")

water_box_bounds = {
  "min_x" : -90.0, "max_x" : -4.0,
  "min_y" : -25.0, "max_y" : 25.0,
  "min_z" : -25.0, "max_z" : 25.0
}

IL_box_bounds = {
  "min_x" : 4.0, "max_x" : 90.0,
  "min_y" : -25.0, "max_y" : 25.0,
  "min_z" : -25.0, "max_z" : 25.0
}

water_box_system = create_water_box(water_box_bounds)
OMIM_PF6_box_system = create_ionic_liquid_box(IL_box_bounds, OMIM_pdb_file_path, PF6_pdb_file_path, 1.24)

graphene_monolayer = cif_file_to_monolayer_membrane(C_cif_file_path, max_dimension=[52, 52, 50])
graphene_monolayer_w_pore = create_pore_circular(graphene_monolayer, 5.0)[0]

graphene_membrane_copy = copy.deepcopy(graphene_monolayer_w_pore)

write_lammps_structure_file_atomic_full(graphene_membrane_copy, "graphene_50by50_membrane_w_pore.data")


graphene_monolayer_w_pore.combine_with_other_structured_system(water_box_system)

graphene_monolayer_w_pore.combine_with_other_structured_system(OMIM_PF6_box_system)

write_lammps_structure_file_atomic_full(graphene_monolayer_w_pore, "H2O_GRAPHENE_OMIMPF6_system.data")
write_pdb_file_from_StructuredSystem(graphene_monolayer_w_pore, "H2O_GRAPHENE_OMIMPF6_system.pdb")
