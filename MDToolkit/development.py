import pandas as pd
from MDToolkit.IO.read_file import read_pdb, read_cif
from MDToolkit.data.objects import *
from MDToolkit.utils.structure_file_utils import get_bounding_box_from_molecule_list, get_bounding_box_angles_from_bounding_box
from MDToolkit.IO.write_file import write_pdb_file_from_StructuredSystem

file_path = '/home/jovinryanj/projects/mdtoolkit/MDToolkit/data/cif_files/MoS2.cif'


read_cif(file_path)
