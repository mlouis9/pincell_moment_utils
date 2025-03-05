from pincell_moment_utils.datagen import DatasetGenerator
from pathlib import Path
import argparse
import numpy as np

np.random.seed(42)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="Whether to generate source files or data. Must be either 'source_file' or 'data'.")
    args = parser.parse_args()

    # This is run by calling `mpiexec --np N_MPI --map-by socket:PE=N_CORES_PER_SOCKET_PER_MPI --bind-to core python mpi_test.py` where
    # N_MPI is the number of MPI processes and N_CORES_PER_SOCKET_PER_MPI is the number of cores per MPI process per socket. If working
    # with a single socket system, then this is just N_CORES_PER_MPI.
    
    num_datapoints = 100
    I = 7
    J = 5
    N_p = 20
    N_w = 9

    generator = DatasetGenerator(num_datapoints, I, J, N_p, N_w, output_dir=Path('./data'), num_histories=int(1E+06), burn_in=1000)

    if args.mode == 'source_file':
        generator.generate_source_files()
    elif args.mode == 'data':
        generator.generate_data()
    else:
        raise ValueError("Invalid argument. Must be either 'source_file' or 'data'.")

if __name__ == '__main__':
    main()
