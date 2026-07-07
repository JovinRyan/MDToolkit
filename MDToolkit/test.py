import os
from MDToolkit.IO.read_file import lammps_data_file_to_structured_system
from MDToolkit.IO.write_file import write_lammps_structure_file_atomic_full
from MDToolkit.data.objects import  Atom, Molecule, StructuredSystem, Simulation
from MDToolkit.paths import  OUTPUT

file_path = OUTPUT
file_name = "Graphene_Water_KCl.data"
file = os.path.join(file_path, file_name)

system = lammps_data_file_to_structured_system(file)
system.populate_angles_from_bonds()
system.populate_elemental_properties_for_all_atoms()

write_lammps_structure_file_atomic_full(system, "Graphene_Water_KCl2.data")