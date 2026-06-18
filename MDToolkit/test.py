# import cProfile
# from MDToolkit.IO.read_file import pdb_file_to_structured_system, lammps_data_file_to_structured_system, lammps_dump_file_to_simulation
# from MDToolkit.IO.write_file import write_lammps_structure_file_atomic_full
# from MDToolkit.analysis.rdf import compute_type_wise_rdf, compute_total_rdf

# lammps_dump_file = "/media/jrjoseph/Elements/projects/training/water_box_ls6/water_box_nvt_prod.out"

# type_map = {1 : "O", 2 : "H"}

# simulation = lammps_dump_file_to_simulation(lammps_dump_file, type_map)

# compute_type_wise_rdf(simulation)

# rdf = compute_total_rdf(simulation)

# print(rdf)

import inspect
from rdfpy import rdf

print(inspect.signature(rdf))