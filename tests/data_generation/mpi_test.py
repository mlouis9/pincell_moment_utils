from pincell_moment_utils.datagen import DatasetGenerator, DefaultPincellParameters
from pathlib import Path
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="Whether to generate source files or data. Must be either 'source_file' or 'data'.")
    args = parser.parse_args()

    # This is run by calling `mpiexec --np N_MPI --map-by socket:PE=N_CORES_PER_SOCKET_PER_MPI --bind-to core python mpi_test.py` where
    # N_MPI is the number of MPI processes and N_CORES_PER_SOCKET_PER_MPI is the number of cores per MPI process per socket. If working
    # with a single socket system, then this is just N_CORES_PER_MPI.

    num_cores_per_proc = len(os.sched_getaffinity(0))  # Get only the cores allocated to the MPI process
    print(f"Cores per proc: {num_cores_per_proc}")
    num_datapoints = 20
    I = 5
    J = 5
    N_p = 5
    N_w = 9
    pincell_params = DefaultPincellParameters()
    pincell_params.wt_enrichment = 0.2
    pincell_params.num_particles_per_generation = int(1E+02)
    generator = DatasetGenerator(num_datapoints, I, J, N_p, N_w, output_dir=Path('./example_data'), burn_in=100,
                                 default_pincell_parameters=pincell_params)

    if args.mode == 'source_file':
        generator.generate_source_files()
    elif args.mode == 'data':
        generator.generate_data()
    else:
        raise ValueError("Invalid argument. Must be either 'source_file' or 'data'.")
    
if __name__ == '__main__':
    main()