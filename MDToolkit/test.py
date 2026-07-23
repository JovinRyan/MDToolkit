import os 
from MDToolkit.IO.read_file import read_lammps_log_file

filedir = "/media/jrjoseph/Elements/projects/smaller_kcl_mos2_ls6/v1/"

filename = "relaxation_job_output.o3310056"

filepath = os.path.join(filedir, filename)

data = read_lammps_log_file(filepath)

print(data)
