import os 
from MDToolkit.data.objects import Topology, Frame, Simulation, LAMMPS_CustomDump_Reader
from MDToolkit.data.misc_objects import BoxVolume
from MDToolkit.utils.structure_file_utils import create_elements_dictionary
from MDToolkit.analysis.density import axial_density, axial_density_time_averaged

filedir = "/media/jrjoseph/Elements/projects/training/graphene_water_kcl_ls6/"
filename = "graphene_water_kcl_2.5V_prod1.out"

filepath = os.path.join(filedir, filename)

type_mapping = {
    1 : "Cl",
    2 : "H",
    3 : "O",
    4 : "C",
    5 : "K",
}

charges_dict = {
    1 : -1.0,
    2 : 0.4238,
    3 : -0.8476,
    4 : 0.0,
    5 : 1.0
}

topol = Topology(type_mapping, elements_dict = create_elements_dictionary(), charges_dict = charges_dict)

reader = LAMMPS_CustomDump_Reader(filepath, topol)

simulation = Simulation(filepath, topol, LAMMPS_CustomDump_Reader)

r = axial_density_time_averaged(simulation, bin_width = 0.25)

import matplotlib.pyplot as plt

def plot_axial_density(
    density_data,
    quantity = "mass_density",
    show_std = True,
    ax = None,
    figsize = (8, 5)
):
    '''
    Plot axial density profiles.

    Parameters
    ----------
    density_data : dict
        Output of axial_density_time_averaged().
    quantity : str
        One of "mass_density", "number_density", or "counts".
    show_std : bool
        Whether to plot ±1 standard deviation.
    '''

    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)

    x = density_data["bin_centers"]

    y = density_data[quantity]["mean"]

    ax.plot(x, y, label = quantity.replace("_", " ").title())

    if show_std:

        std = density_data[quantity]["std"]

        ax.fill_between(
            x,
            y - std,
            y + std,
            alpha = 0.3
        )

    xlabel = ["x", "y", "z"][0]

    ax.set_xlabel(f"{xlabel} Position (Å)")
    ax.set_ylabel(quantity.replace("_", " ").title())
    ax.legend()

    return ax

plot_axial_density(r)
plt.show()