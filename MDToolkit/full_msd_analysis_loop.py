import os 
import numpy as np
from MDToolkit.data.objects import Topology, Frame, Simulation, LAMMPS_CustomDump_Reader
from MDToolkit.utils.structure_file_utils import create_elements_dictionary
from MDToolkit.utils.misc_utils import get_n_even_chunks
from MDToolkit.analysis.diffusivity import compute_msd, compute_diffusivity
from MDToolkit.IO.write_file import export_numeric_data

molarity = 1
ensembles = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for ensemble in ensembles:

    # filedir = f"/media/jrjoseph/Elements/projects/fem_memristor/md_simulations/cscl_diffusivity_ls6/{molarity}M/"
    filedir = f"/media/jrjoseph/Elements/projects/fem_memristor/md_simulations/cscl_{molarity}M_ls6/v{ensemble}"
    filename = f"cscl_{molarity}M_water_prod{ensemble}.out"

    filepath = os.path.join(filedir, filename)

    type_mapping = {
        1 : "C",
        2 : "Cl",
        3 : "H",
        4 : "Cs",
        5 : "O",
    }

    topol = Topology(type_mapping, elements_dict = create_elements_dictionary())

    simulation = Simulation(filepath, topol, LAMMPS_CustomDump_Reader, cpp_init=False)

    ion_spcs = ["Cs", "Cl"]

    msd_data = compute_msd(simulation, ion_spcs)

    export_numeric_data(msd_data, f"CsCl_{molarity}M_run{ensemble}_MSD.csv", file_path=filedir + "/../")