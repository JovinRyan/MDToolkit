import os 
from MDToolkit.data.objects import Topology, Frame, Simulation, LAMMPS_CustomDump_Reader, MultiSimulation
from MDToolkit.utils.structure_file_utils import create_elements_dictionary
from MDToolkit.analysis.current import qv_current, qv_current_time_averaged, ion_translocation
from MDToolkit.data.misc_objects import  BoxVolume, CylinderVolume
from MDToolkit.IO.read_file import read_lammps_log_file
import pandas as pd
from MDToolkit.IO.write_file import export_numeric_data

runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
frequency = 200 # MHz

for run in runs:
    times = [20]

    filedir = f"/media/jrjoseph/Elements/projects/fem_memristor/md_simulations/cscl_graphene_{frequency}MHz_ls6/v{run}"
    filenames = [f"cscl_graphene_prod_{time}ns_run{run}.out" for time in times]

    filepaths = [os.path.join(filedir, filename) for filename in filenames]

    type_mapping = {
        1 : "C",
        2 : "Cl",
        3 : "Cs",
        4 : "H",
        5 : "O",
    }

    charges_dict = {
        1 : 0.0,
        2 : -1.0,
        4 : 0.4238,
        3 : 1.0,
        5 : -0.8476
    }

    ion_spcs = ["Cs", "Cl"]

    topol = Topology(type_mapping, elements_dict = create_elements_dictionary(), charges_dict = charges_dict)

    # simulation = Simulation(filepath, topol, LAMMPS_CustomDump_Reader)

    simulation = MultiSimulation(filepaths, topol, LAMMPS_CustomDump_Reader, cpp_init = False)

    v1 = BoxVolume([-50, -25, -25], [-1, 25, 25])
    v_trans = CylinderVolume([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], radius = 3.5)
    v2 = BoxVolume([1, -25, -25], [50, 25, 25])

    translocations = ion_translocation(simulation, ion_spcs, v1, v_trans, v2)

    flat = pd.json_normalize(translocations, sep="_").to_dict(orient="records")[0]

    export_numeric_data(flat, f"CsCl_{frequency}MHz_run{run}_{max(times)}ns_translocations.csv", file_path=f"/media/jrjoseph/Elements/projects/fem_memristor/md_simulations/cscl_graphene_{frequency}MHz_ls6/")
