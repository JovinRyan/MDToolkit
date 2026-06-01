import pandas as pd
import os
from MDToolkit.custom_systems.liquids import create_water_box, create_ionic_liquid_box
from MDToolkit.custom_systems.membranes import cif_file_to_monolayer_membrane
from MDToolkit.IO.write_file import write_lammps_structure_file_atomic_full
from MDToolkit.IO.read_file import pdb_file_to_structured_system
from MDToolkit.paths import CIF_FILES

C_cif_file_path = os.path.join(CIF_FILES, "graphene.cif")
MoS2_cif_file_path = os.path.join(CIF_FILES, "MoS2.cif")

graphene_monolayer = cif_file_to_monolayer_membrane(C_cif_file_path, max_dimension=[50, 50, 50])

MoS2_monolayer = cif_file_to_monolayer_membrane(MoS2_cif_file_path, max_dimension=[50, 50, 50])

write_lammps_structure_file_atomic_full(graphene_monolayer, "graphene_50by50_membrane.data")
write_lammps_structure_file_atomic_full(MoS2_monolayer, "MoS2_50by50_membrane.data")
