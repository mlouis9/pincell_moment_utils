from pincell_moment_utils.datagen import DatasetGenerator
from pathlib import Path
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("mode", type=str, help="Whether to generate source files or data. Must be either 'source_file' or 'data'.")
args = parser.parse_args()

# This is run by calling `mpiexec --np N_MPI --map-by socket:PE=N_CORES_PER_SOCKET_PER_MPI --bind-to core python test.py` where
# N_MPI is the number of MPI processes and N_CORES_PER_SOCKET_PER_MPI is the number of cores per MPI process per socket. If working
# with a single socket system, then this is just N_CORES_PER_MPI.

num_cores_per_proc = len(os.sched_getaffinity(0))  # Get only the cores allocated to the MPI process
print(f"Cores per proc: {num_cores_per_proc}")
num_datapoints = 10
I = 5
J = 5
N_p = 5
N_w = 4
generator = DatasetGenerator(num_datapoints, I, J, N_p, N_w, output_dir=Path('./example_data'), num_histories=int(1E+04), burn_in=100)

if args.mode == 'source_file':
    generator.generate_source_files()
elif args.mode == 'data':
    generator.generate_data()
else:
    raise ValueError("Invalid argument. Must be either 'source_file' or 'data'.")