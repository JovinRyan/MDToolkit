from MDToolkit.IO.read_file import lammps_data_file_to_structured_system, lammps_dump_file_to_simulation
from MDToolkit.data.objects import Simulation, StructuredSystem
from MDToolkit.data.misc_objects import BoxVolume, CylinderVolume, get_max_box_volume_from_simulation
from MDToolkit.analysis.current import charge_velocity_current
import matplotlib.pyplot as plt

cyl = CylinderVolume([-24.5, 0, 0], [24.5, 0, 0], 10.5)

file_path = "/media/jrjoseph/Elements/projects/training/graphene_water_kcl_ls6/graphene_water_kcl_2.5V_prod1.out"

type_dict = {
    1 : "Cl",
    2 : "H",
    3 : "O",
    4 : "C",
    5 : "K"
}

simulation = lammps_dump_file_to_simulation(file_path, type_dict)

I = charge_velocity_current(simulation)

print(I)

plt.plot(simulation.timesteps, I)
plt.show()
