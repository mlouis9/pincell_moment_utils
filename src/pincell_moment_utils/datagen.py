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
import subprocess
import shutil
import zarr

# Get the absolute path to the directory containing this file
file_path = Path(__file__).resolve()
file_directory = file_path.parent

class DefaultPincellParameters:
    """This class is used for passing information used to template the default pincell calculation script."""
    def __init__(self, wt_enrichment: float=0.04, fuel_density: float=10.0, water_density: float=1.0, pitch: float=1.26, fuel_or: float=0.39,
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
        fuel_or
            The outer radius of the fuel pin in cm (make sure this is consistent with the pitch)
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
        self.fuel_or = fuel_or

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
            fuel_or=self.fuel_or,
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
                 energy_filters: list=DefaultPincellParameters().energy_filters, 
                 num_histories: int=DefaultPincellParameters().num_histories(), zernlike_order: int=DefaultPincellParameters().zernlike_order,
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
        zernlike_order
            The order of the Zernike expansion to use in the pincell calculation (this is used to determine the number of sample particles to generate
            for each randomly sampled incident flux profile). Note zernlike_order is not required if default_pincell_parameters is specified, as
            the order of the Zernike expansion is already defined in the DefaultPincellParameters class.
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

        Notes
        -----
        - If num_datapoints is greater than the number of unique assignments of surface weights and surface profiles, then only a number of points equal to the number of 
            unique assignments will be generated.
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
        self.zernlike_order = zernlike_order

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
                    default_pincell_parameters = DefaultPincellParameters()
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

        
    def generate_samples(self) -> None:
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

        cores_per_proc = len(os.sched_getaffinity(0))
        cores_per_proc = int(os.environ.get("OMP_NUM_THREADS", cores_per_proc)) # Override if OMP_NUM_THREADS is set

        nexp = len(self.expansions)

        for i in range(rank, nexp, size):
            print(f"Rank {rank} generating samples for expansion {i}")
            expansion = self.expansions[i]
            # We'll generate 4*self.num_histories samples, because each expansion has 4 surfaces
            big_chunk = expansion.generate_samples(self.num_histories * 4, burn_in=self.burn_in,
                                                   progress=True, num_cores=cores_per_proc)
            big_chunk = np.array(big_chunk)
            
            start_idx = i * 4
            end_idx = start_idx + 4

            # If the last expansion is only partially used (N_p not multiple of 4), adjust
            if i == nexp - 1 and (self.N_p % 4) != 0:
                remainder = self.N_p % 4
                end_idx = start_idx + remainder
                big_chunk = big_chunk[: remainder, :, :]

            # Reshape into (n_surfaces_for_this_expansion, num_histories, 3)
            num_surfaces = end_idx - start_idx
            reshaped = big_chunk.reshape(num_surfaces, self.num_histories, 3)

            # Write each surface to its own file
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True, exist_ok=True)

            for j in range(num_surfaces):
                surface_id = start_idx + j
                with h5py.File(self.output_dir / f"profile_{surface_id}.h5", 'w') as hf:
                    hf.create_dataset("samples", data=reshaped[j], compression="gzip")

        comm.Barrier()


    def generate_data(self):
        """Generate the data for each case in `self.assignments` using the source files generated
        via the `generate_source_files` method. The data is written to the output directory specified 
        in the DatasetGenerator class. This implementation has each MPI rank write its own temporary 
        file and rank 0 later merges these into a final .zarr store.
        """

        comm = self.MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # Get dimensions for the final arrays
        G = len(self.energy_filters[0].values) - 1
        flux_dim = (4, self.I, self.J, G, 1)
        zernike_dim = (self.zernlike_order + 1) * (self.zernlike_order + 2) // 2

        # Prepare dictionaries to store local results
        local_indices = []
        local_X_flux = []
        local_X_wts = []
        local_Y_flux = []
        local_Y_pow = []
        local_Y_keff = []
        local_Y_wts = []

        # Prepare environment for running each pincell case
        env_no_mpi = {
            'PATH': os.environ['PATH'],
            'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', ''),
            'HOME': os.environ['HOME'],
            'OPENMC_CROSS_SECTIONS': os.environ.get('OPENMC_CROSS_SECTIONS', ''),
            "PYTHONPATH": os.environ["PYTHONPATH"],
            "TMPDIR": str(self.output_dir),  # Force temporary files to be created in output_dir
        }

        # Loop over assigned data points
        for index in range(rank, len(self.assignments), size):
            # Each assignment is (weights, surface_assignments)
            assignment = self.assignments[index]
            weights = np.array(assignment[0])
            surface_assignments = assignment[1]
            N_surface = round_preserving_sum(weights * self.num_histories)
            nonzero_surfaces = [
                surface for surface in range(4)
                if surface_assignments[surface] is not None
            ]

            # Build the ephemeral source file
            source_particles = []
            for surface in nonzero_surfaces:
                expansion_idx = surface_assignments[surface]  # which row in the big samples array
                if expansion_idx is None:
                    continue
                
                # Each file "profile_{surface_id}.h5" has shape (num_histories, 3)
                surface_id = surface_assignments[surface]
                profile_path = self.output_dir / f"profile_{surface_id}.h5"
                if not profile_path.is_file():
                    raise FileNotFoundError(
                        f"Expected file {profile_path} not found. "
                        "Make sure generate_samples() completed successfully."
                    )
                
                # Read only as many samples as needed for this surface
                with h5py.File(profile_path, 'r') as pf:
                    chunk = pf["samples"][:N_surface[surface], :]
                
                # Transform angles for the appropriate surface
                angle_transform = config.INCIDENT_ANGLE_TRANSFORMATIONS[
                    assignment[1][surface] % 4
                ][surface]

                # Convert sample coordinate from config
                surface_coord_3d = config.SURFACE_COORD_TO_3D[surface]

                for sample in chunk:
                    xval, omega_in, E = sample
                    omega_transformed = angle_transform(omega_in)
                    p = openmc.SourceParticle(
                        r=surface_coord_3d(xval),
                        u=(np.cos(omega_transformed), np.sin(omega_transformed), 0.0),
                        E=E
                    )
                    source_particles.append(p)

            # Write to a temporary source file
            source_file_path = self.output_dir / f"source{index}.h5"
            openmc.write_source_file(source_particles, source_file_path)

            # ====== Add Metadata to the Source File ======
            with h5py.File(self.output_dir / f"source{index}.h5", 'a') as f:
                f.attrs["nonzero_surfaces"] = np.array(nonzero_surfaces, dtype=np.int32)

            # Run the pincell calculation with this temp source file
            proc = subprocess.Popen(
                [
                    'python', str(self.pincell_path), 
                    str(source_file_path), 
                    '--outdir', str(self.output_dir)
                ],
                env=env_no_mpi
            )
            proc.communicate()

            # The pincell script is expected to produce statepoint.source{index}.h5
            sp_path = self.output_dir / f"statepoint.source{index}.h5"

            # Now post-process
            mesh_tally = pp.SurfaceMeshTally(str(sp_path))
            coefficients = pp.compute_coefficients(mesh_tally, self.I, self.J)
            coefficients = normalize_outgoing_coefs(
                coefficients, self.energy_filters, self.I, self.J
            )

            # Grab the Zernike / keff / power weighting from the statepoint
            with openmc.StatePoint(sp_path) as sp:
                df = sp.get_tally(name='zernike').get_pandas_dataframe()

                # Example: gather some placeholders for demonstration
                # (your real logic to parse tallies might differ)
                tally_names = [(id, sp.tallies[id].name) for id in sp.tallies.keys()]
                nonleakage_tally_ids = [
                    tid for tid, name in tally_names if 'nonleakage' in name
                ]
                nonleakage_tallies = np.array([
                    sp.get_tally(id=tid).mean for tid in nonleakage_tally_ids
                ])
                keff = np.sum(np.abs(nonleakage_tallies))

                # Example: parse some weighting from tallies with IDs 6..9
                y_wts = []
                for tid in range(6, 10):
                    if tid in sp.tallies:
                        y_wts.append(sp.get_tally(id=tid).mean[0][0][0])
                    else:
                        y_wts.append(0.0)
                y_wts = np.abs(y_wts)
                if np.sum(y_wts) > 0:
                    y_wts /= np.sum(y_wts)

            means = df['mean']
            std_devs = df['std. dev.']
            filtered_coeffs = [
                mean if abs(mean) >= std_dev else 0
                for mean, std_dev in zip(means, std_devs)
            ]
            filtered_coeffs = np.array(filtered_coeffs).reshape(zernike_dim, G)

            # Gather input data for this assignment
            weights = np.array(self.assignments[index][0])
            surface_indices = self.assignments[index][1]
            surface_expansions = [self.expansions[index // 4].coefficients[index % 4] 
                                if index is not None else np.zeros(flux_dim[1:]) 
                                for index in surface_indices]
            surface_expansions = np.array(surface_expansions)

            # Append local results
            local_indices.append(index)
            local_X_flux.append(surface_expansions)   # or real expansions if you are storing them
            local_X_wts.append(weights)
            local_Y_flux.append(coefficients)
            local_Y_pow.append(filtered_coeffs)
            local_Y_keff.append(keff)
            local_Y_wts.append(y_wts)

            # Delete temporary source and statepoint files to minimize storage
            if source_file_path.exists():
                source_file_path.unlink()
            if sp_path.exists():
                sp_path.unlink()

        # Save local results to a temporary .npz
        temp_filename = self.output_dir / f"temp_data_rank_{rank}.npz"
        np.savez(
            temp_filename,
            indices=np.array(local_indices),
            X_flux=np.array(local_X_flux),
            X_wts=np.array(local_X_wts),
            Y_flux=np.array(local_Y_flux),
            Y_pow=np.array(local_Y_pow),
            Y_keff=np.array(local_Y_keff),
            Y_wts=np.array(local_Y_wts)
        )

        comm.Barrier()

        # Rank 0 merges everything into a Zarr store
        if rank == 0:
            store_path = self.output_dir / "dataset.zarr"
            store = zarr.storage.LocalStore(store_path)
            root = zarr.group(store=store, overwrite=True)
            flux_chunks = (1000, 4, self.I, self.J, G, 1)

            X_flux_array = root.create_array(
                "X_flux_coeffs", 
                shape=(self.num_datapoints, *flux_dim),
                chunks=flux_chunks,
                dtype='float64'
            )
            X_wts_array = root.create_array(
                "X_weights",
                shape=(self.num_datapoints, 4),
                chunks=(self.num_datapoints, 4),
                dtype='float64'
            )
            Y_flux_array = root.create_array(
                "Y_flux_coeffs", 
                shape=(self.num_datapoints, *flux_dim),
                chunks=flux_chunks,
                dtype='float64'
            )
            Y_pow_array = root.create_array(
                "Y_power_coeffs", 
                shape=(self.num_datapoints, zernike_dim, G),
                chunks=(self.num_datapoints, zernike_dim, G),
                dtype='float64'
            )
            Y_keff_array = root.create_array(
                "Y_keff", 
                shape=(self.num_datapoints,),
                chunks=(self.num_datapoints,),
                dtype='float64'
            )
            Y_wts_array = root.create_array(
                "Y_weights", 
                shape=(self.num_datapoints, 4),
                chunks=(self.num_datapoints, 4),
                dtype='float64'
            )

            # Merge data from all ranks
            for r in range(size):
                temp_file = self.output_dir / f"temp_data_rank_{r}.npz"
                if temp_file.exists():
                    data = np.load(temp_file)
                    indices = data["indices"]
                    X_flux_data = data["X_flux"]
                    X_wts_data = data["X_wts"]
                    Y_flux_data = data["Y_flux"]
                    Y_pow_data = data["Y_pow"]
                    Y_keff_data = data["Y_keff"]
                    Y_wts_data = data["Y_wts"]

                    for local_idx, global_index in enumerate(indices):
                        X_flux_array[global_index] = X_flux_data[local_idx]
                        X_wts_array[global_index] = X_wts_data[local_idx]
                        Y_flux_array[global_index] = Y_flux_data[local_idx]
                        Y_pow_array[global_index] = Y_pow_data[local_idx]
                        Y_keff_array[global_index] = Y_keff_data[local_idx]
                        Y_wts_array[global_index] = Y_wts_data[local_idx]

                    os.remove(temp_file)
                else:
                    print(f"Warning: Temporary file for rank {r} not found.")

        comm.Barrier()


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


def normalize_outgoing_coefs(coefficients, energy_filters, n_spatial_terms, n_angular_terms):
    for surface in range(4):
        smin, smax = config.SPATIAL_BOUNDS[surface]
        omin, omax = config.OUTGOING_ANGULAR_BOUNDS[surface]

        # 2) Pull the energy bin widths
        efilter = energy_filters[surface]
        dE = np.diff(efilter.bins, axis=1).flatten()  # shape (N_energy,)

        # 3) The total scale factor for space+angle:
        #    Because integral of B_{i,I-1} is 1/I on [0,1],
        #    after scaling to [smin, smax], we get (smax-smin)/I.
        #    Similarly (omax-omin)/J for the angle dimension.
        space_factor = ((smax - smin)/n_spatial_terms) * \
                    ((omax - omin)/n_angular_terms)

        # 4) Our coefficients for this surface: shape (I, J, N_energy, 1)
        c_ijE = coefficients[surface, :, :, :, 0]  # => shape (I, J, N_energy)

        # 5) Sum over i,j => sum_c[k] = Σ_{i=0..I-1} Σ_{j=0..J-1} c_{i,j,k}
        sum_c_k = c_ijE.sum(axis=(0,1))  # shape (N_energy,)

        # 6) Multiply each sum_c_k by (space_factor * dE[k]) and sum over k
        flux_integral = np.sum(sum_c_k * space_factor * dE)
        # 7) Normalize the coefficients
        coefficients[surface, :, :, :, 0] /= flux_integral
    
    return coefficients