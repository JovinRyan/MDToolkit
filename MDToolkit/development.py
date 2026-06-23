from MDToolkit.IO.read_file import lammps_data_file_to_structured_system
from MDToolkit.data.objects import StructuredSystem
from MDToolkit.data.misc_objects import BoxVolume, CylinderVolume
from MDToolkit.analysis.density import axial_density_new

system_file_path = "/media/jrjoseph/Elements/projects/training/cnt_graphene_water_ls6/CNT_Graphene_Water_filled.data"

system = lammps_data_file_to_structured_system(system_file_path)

volumes = [BoxVolume([-104.027, -27.70, -27.155], [-53, 27.70, 27.155]), CylinderVolume([-53, 0, 0], [53, 0, 0], 7.5), BoxVolume([53, -27.70, -27.155], [104.011, 27.70, 27.155])]

out = axial_density_new(system, volumes, bins=500)

import matplotlib.pyplot as plt

def plot_axial_density(
    density_data,
    quantity="density",
    ax=None,
    **plot_kwargs
):
    '''
    Parameters
    ----------
    density_data : dict
        Output from axial_density() or axial_density_new()

    quantity : str
        "density"
        "number_density"

    ax : matplotlib axis, optional
    '''

    if ax is None:
        fig, ax = plt.subplots()

    x = density_data["bin_centers"]
    y = density_data[quantity]

    ax.plot(x, y, **plot_kwargs)

    ax.set_xlabel("Position")
    ax.set_ylabel(quantity.replace("_", " ").title())

    if quantity == "density":
        ax.axhline(
            density_data["average_density"],
            linestyle="--",
            label="Average Density"
        )
        ax.legend()

    return ax

plot_axial_density(out)
plt.show()