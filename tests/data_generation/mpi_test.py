from pincell_moment_utils.datagen import DatasetGenerator
from pathlib import Path
import os

# This is run by calling `mpiexec --np N_MPI --map-by socket:PE=N_CORES_PER_SOCKET_PER_MPI --bind-to core python test.py` where
# N_MPI is the number of MPI processes and N_CORES_PER_SOCKET_PER_MPI is the number of cores per MPI process per socket. If working
# with a single socket system, then this is just N_CORES_PER_MPI.

num_cores_per_proc = len(os.sched_getaffinity(0))  # Get only the cores allocated to the MPI process
print(f"Cores per proc: {num_cores_per_proc}")
num_datapoints = 100
I = 5
J = 5
N_p = 5
N_w = 4
DatasetGenerator(num_datapoints, I, J, N_p, N_w, output_dir=Path('./example_data'), num_histories=int(1E+05), burn_in=1000)