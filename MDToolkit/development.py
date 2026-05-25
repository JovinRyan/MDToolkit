import pandas as pd
from MDToolkit.IO.read_file import read_pdb, read_cif, cif_file_to_structured_system
from MDToolkit.data.objects import *
from MDToolkit.utils.structure_file_utils import create_periodic_images
from MDToolkit.IO.write_file import write_pdb_file_from_StructuredSystem

file_path_MoS2 = '/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/cif_files/MoS2.cif'

MoS2_structured_system = cif_file_to_structured_system(file_path_MoS2)

combined_system = create_periodic_images(MoS2_structured_system, (8, 8, 2))

write_pdb_file_from_StructuredSystem(combined_system, file_name="MoS2_8x8x2.pdb", file_path="./Output")
