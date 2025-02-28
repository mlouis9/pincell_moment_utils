"""This module contains the functions necessary for generating the dataset of incident flux/outgoing flux cases.

Describe the dataset generation process here: TODO

Describe assumptions about the pincell calculation file here: TODO"""

from pincell_moment_utils import postprocessing as pp
from pathlib import Path, PosixPath, WindowsPath
from typing import Union, List
from tempfile import NamedTemporaryFile
from string import Template
import openmc
import h5py
import numpy as np
from math import ceil
from pincell_moment_utils.sampling import generate_all_unique_assignments
import os
from pincell_moment_utils import config

# Get the absolute path to the directory containing this file
file_path = Path(__file__).resolve()
file_directory = file_path.parent

class DefaultPincellParameters:
    """This class is used for passing information used to template the default pincell calculation script."""
    def __init__(self, wt_enrichment: float=0.04, fuel_density: float=10.0, water_density: float=1.0, pitch: float=1.26,
                 num_batches: int=100, num_inactive: int=10, num_particles_per_generation: int=int(1E+04), 
                 zernlike_order: int=4, N: int=40, N_omega: int=20, energy_file: Union[Path, str]=file_directory / 'data' / 'cas8.h5'):
        """Define the parameters used to template the default pincell calculation script.
        
        Parameters
        ----------
        wt_enrichment
            The weight percent enrichment of the fuel
        fuel_density
            The density of the fuel in g/cc
        water_density
            The density of the water in g/cc
        pitch
            The pitch of the fuel pin in cm
        num_batches
            The number of batches to use in the pincell calculation
        num_inactive
            The number of inactive batches to use in the pincell calculation
        num_particles_per_generation
            The number of particles to use per generation in the pincell calculation
        zernlike_order
            The order of the Zernike expansion to use in the pincell calculation
        N
            The number of mesh points to use along each surface in the pincell calculation
        N_omega
            The number of angular bins to use in the pincell calculation
        energy_file
            - The path to the energy file to use in the pincell calculation
            - Or a string used to identify one of the default energy files to use in the pincell calculation, currently 'cas8', 'cas14',
              'cas25', and 'cas70' are supported, corresponding to the CASMO 8, 14, 25, and 70 group structures, respectively.
        """
        self.wt_enrichment = wt_enrichment
        self.fuel_density = fuel_density
        self.water_density = water_density
        self.pitch = pitch
        self.num_batches = num_batches
        self.num_inactive = num_inactive
        self.num_particles_per_generation = num_particles_per_generation
        self.zernlike_order = zernlike_order
        self.N = N
        self.N_omega = N_omega

        if type(energy_file) == str:
            self.energy_file = file_directory / 'data' / f'{energy_file}.h5'
        elif type(energy_file) == PosixPath or type(energy_file) == WindowsPath:
            self.energy_file = energy_file
        else:
            raise TypeError(f"energy_file must be a string or a Path object, not {type(energy_file)}")
        
        self.energy_filters = self.get_energy_filters()
        
    def num_histories(self):
        """The total number of histories used in the pincell calculation."""
        return self.num_particles_per_generation * self.num_batches
    
    def get_energy_filters(self):
        """Read the energy file and return the energy filters used in the pincell calculation."""
        with h5py.File(self.energy_file, 'r') as f:
            energy_groups = f['energy groups'][:]
        return [openmc.EnergyFilter(energy_groups) for surface in range(4) ]
    
    def create_script(self, script_path: Path=None) -> Path:
        """Create a script using the parameters defined in this class.
        
        Parameters
        ----------
        script_path
            The path to the script to create. If None, a temporary file is created and the path to the temporary file is returned.
            
        Returns
        -------
        script_path
            The path to the script created. If script_path is None, a temporary file is created and the path to the temporary file is returned.
        """

        # First read and template the default pincell calculation script
        # Then write the script to the specified path

        with open(file_directory / 'input_files' / 'pincell.py', 'r') as f:
            script_template = Template(f.read())
            
        templated_script = script_template.safe_substitute(
            wt_enrichment=self.wt_enrichment,
            fuel_density=self.fuel_density,
            water_density=self.water_density,
            pitch=self.pitch,
            num_batches=self.num_batches,
            num_inactive=self.num_inactive,
            num_particles_per_generation=self.num_particles_per_generation,
            zernlike_order=self.zernlike_order,
            N=self.N,
            N_omega=self.N_omega,
            energy_file=str(self.energy_file.resolve()),
        )

        if script_path is None:
            with NamedTemporaryFile(delete=False, suffix='.py') as f:
                f.write(templated_script.encode('utf-8'))
                script_path = Path(f.name)
            return script_path
        else:
            with open(script_path, 'w') as f:
                f.write(templated_script)

class DatasetGenerator:
    """This class is used for generating the dataset of incident flux/outgoing flux cases."""
    def __init__(self, num_datapoints: int, I: int, J: int, N_p: int, N_w: int, output_dir: Path, 
                 energy_filters: list=DefaultPincellParameters().get_energy_filters(), 
                 num_histories: int=DefaultPincellParameters().num_histories(), 
                 pincell_path: Path=None, default_pincell_parameters: DefaultPincellParameters=None,
                 burn_in: int=1000):
        """Generate the dataset of incident flux/outgoing flux cases for a given pincell calculation.
        
        Parameters
        ----------
        num_datapoints
            The total number of datapoints to generate for the dataset (this is the number of incident flux/outgoing flux cases)
        I
            The number of spatial expansion functions (equal to the spatial expansion order - 1) to use when generating the dataset
        J
            The number of angular expansion functions (equal to the angular expansion order - 1) to use when generating the dataset
        N_p
            The number of incident flux profiles to randomly sample when parameterizing the coefficient space for the boundary conditions
        N_w
            The number of values each surface weight can take on (this is used to generate a deterministic interpolation of possible
            surface weights for the boundary conditions)
        output_dir
            The directory to write the dataset to. If the directory does not exist, it will be created.
        energy_filters
            List of openmc EnergyFilter objects used to define the energy groups for the dataset
        num_histories
            The number of histories used in the pincell calculation (this is used to determine the number of sample particles to generate
            for each randomly sampled incident flux profile). Note num_histories is not required if default_pincell_parameters is specified, as
            the number of histories is already defined in the DefaultPincellParameters class.
        pincell_path
            The path to the pincell calculation file (this can be your own, so long as you verify that the correct tallies are implemented
            and in the correct manner). If None, a default pincell script is used (which may nonetheless be useful for most LWR applications),
            which can be templated via the parameters in the DefaultPincellParameters class.
        default_pincell_parameters
            The parameters used to template the default pincell calculation script (must be an instance of the DefaultPincellParameters class). 
            If not specified, the default parameters are used. These parameters are defined in the DefaultPincellParameters class.
        burn_in
            The number of particles to use for the burn-in period in the surface flux sampling.
            
        Returns
        -------
        """
        from mpi4py import MPI

        self.MPI = MPI # Save the imported module as an instance variable so that it can be used in other methods of the class

        self.num_datapoints = num_datapoints
        self.I = I
        self.J = J
        self.N_p = N_p
        self.N_w = N_w
        self.output_dir = output_dir
        self.burn_in = burn_in

        # --------------------------------
        # Pincell parameter processing
        # --------------------------------
        if pincell_path is None:
            pincell_path = file_directory / 'input_files' / 'pincell.py'
            if default_pincell_parameters is not None:
                self.num_histories = default_pincell_parameters.num_histories()
            else: # default_pincell_parameters not supplied
                if num_histories != DefaultPincellParameters().num_histories(): # A non-default number of histories was specified
                    if num_histories % DefaultPincellParameters().num_batches != 0:
                        raise ValueError(f"num_histories must be a multiple of num_batches ({default_pincell_parameters.num_batches})"
                                         "unless default_pincell_parameters is specified.")
                    default_pincell_parameters = DefaultPincellParameters()
                    default_pincell_parameters.num_particles_per_generation = num_histories // default_pincell_parameters.num_batches
                    self.num_histories = num_histories
                else: # num_histories is the default value
                    self.num_histories = default_pincell_parameters.num_histories()

            # Create templated script in a temporary file
            self.pincell_path = default_pincell_parameters.create_script()
            self.energy_filters = default_pincell_parameters.energy_filters

        else: # User specified their own pincell calculation
            self.num_histories = num_histories
            self.pincell_path = pincell_path
            if default_pincell_parameters is not None:
                raise ValueError("default_pincell_parameters cannot be specified if pincell_path is specified.")
            
            self.energy_filters = energy_filters

        # -------------------------------------------
        # Generate Random Expansions and Assignments
        # -------------------------------------------
        # Note we generate 4 profiles per surface expansion, so we divide by 4 to get the correct number of profiles
        self.expansions = [ pp.random_surface_expansion(self.I, self.J, self.energy_filters, incident=True) for _ in range(ceil(self.N_p/4)) ]
        self.assignments = generate_all_unique_assignments(self.N_p, self.N_w, num_samples=self.num_datapoints)

        # Parse the assignments so that it is a list of pairs of weights and surface assignments
        self.assignments = [ ( assignment[0], surface_assignment) for assignment in self.assignments for surface_assignment in assignment[1] ]
        print(len(self.assignments))

        # ---------------
        # Begin sampling
        # ---------------
        all_samples = self.generate_samples()

        # now, use assignments to construct source files for each case in the dataset, labeling them properly
        self.generate_source_files(all_samples)
        
    def generate_samples(self) -> np.ndarray:
        """Generate samples for each of the randomly generated surface expansions. 
        
        Returns
        -------
        samples
            A numpy array corresponding to the samples generated for each surface of each randomly generated expansion. Note the samples
            corespond to the surfce whose index is given by the modulo of the first index of the output array by 4. 
        """

        comm = self.MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Get number of OpenMP threads per process
        omp_threads = int(os.environ.get("OMP_NUM_THREADS", 1))

        all_samples = np.zeros((self.N_p, self.num_histories, 3))
        for index, expansion in enumerate(self.expansions):
            samples = expansion.generate_samples(self.num_histories*4, num_cores=omp_threads, burn_in=self.burn_in, progress=True)
            if index == len(self.expansions) - 1:
                samples = samples[:self.N_p % 4]
                all_samples[index*4:(index*4 + self.N_p % 4)] = samples
            else:
                all_samples[index*4:(index+1)*4] = samples

        comm.Barrier()

        return all_samples

    def generate_source_files(self, all_samples: np.ndarray) -> None:
        """Generate source files for each case in `self.assignments` using the samples (of the randomly generated surface profiles).
        These source files are written to the output directory specified in the DatasetGenerator class.
        
        Parameters
        ----------
        all_samples
            A numpy array of samples generated for each surface of each randomly generated expansion. Note the samples
            corespond to the surfce whose index is given by the modulo of the first index of the output array by 4. 
        """

        # First, create the output directory if it does not exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        
        surface_coord_to_3d = config.SURFACE_COORD_TO_3D
        incident_angle_transformations = config.INCIDENT_ANGLE_TRANSFORMATIONS

        # Now, generate the source files for each case in self.assignments
        for index, assignment in enumerate(self.assignments):
            source_particles = []

            # First extract the surface weights and the surface assignments from the assignment
            weights = np.array(assignment[0])
            N_surface = round_preserving_sum(weights * self.num_histories)
            surface_assignments = assignment[1]
            nonzero_surfaces = [surface for surface in range(4) if surface_assignments[surface] is not None] # Surfaces that have non-zero weights
            
            # First append source particles to an OpenMC SourceParticle object for each surface in the assignment
            for surface in nonzero_surfaces:
                # First transform the angular variable to the appropriate angular domain for the surface that the profile is being
                # assigned to. This is done by using the incident angle transformations defined in the config module.
                original_surface = assignment[1][surface] % 4
                angle_transform = incident_angle_transformations[original_surface][surface]

                for sample in all_samples[assignment[1][surface], :N_surface[surface]]: # Only draw N_surface samples from each surface
                    ω = angle_transform(sample[1]) # Transform the angular domain to that for the appropriate surface

                    # Now, create the source particle for the surface
                    p = openmc.SourceParticle(r=surface_coord_to_3d[surface](sample[0]), u = (np.cos(ω), np.sin(ω), 0), 
                                          E=sample[2])
                    source_particles.append(p)

            # Now write the samples to a source file
            openmc.write_source_file(source_particles, f"source{index}.h5")


def round_preserving_sum(arr: np.ndarray) -> np.ndarray:
    """For a numpy float array that sums to an integet (i.e. multiplying a known number of samples by a weight array whose entries sum
    to 1), round the array to integers such that the sum of the rounded array is equal to the sum of the original array. This is used to
    generate the number of particles to sample for each surface in the dataset, such that the total number of particles sampled is equal
    to the total number of particles used in the pincell calculation.
    
    Parameters
    ----------
    arr
        A numpy array of floats that sums to an integer (i.e. the number of particles to sample for each surface in the dataset)

    Returns
    -------
    arr
        A numpy array of integers that sums to the same integer as the original array (i.e. the number of particles to sample for each surface 
        in the dataset)
    """
    total_original = round(np.sum(arr))  # Ensure target sum is an integer
    floored = np.floor(arr)  # Round everything down
    deficit = int(total_original - np.sum(floored))  # How many elements need rounding up

    # Sort indices based on fractional remainders (largest first)
    remainders = arr - floored
    indices = np.argsort(remainders)[::-1]  # Sort in descending order of remainder

    # Create the final array, rounding up the necessary elements
    result = floored.copy()
    result[indices[:deficit]] += 1  # Increase elements with largest remainders

    return result.astype(int)  # Ensure integer output
