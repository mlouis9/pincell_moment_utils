r"""
This is a module for postprocessing results of the pincell simulation, namely mesh tallies which are used to compute functional expansion
moments, as well as tallied moments for zernlike expansions.

------------
Assumptions
------------
1. The pincell domain is a 2D pitch×pitch square, and the tally regions are (pitch/4)×pitch rectangular (oriented in different directions) void
regions that lie just beyond the pitch×pitch square that bounds the moderator. I.e. the geometry is as below

                   y
                   ^        pitch/2
                   |           |
_______________________________________ _ _ _ _ _ 3/4 pitch
|      |                       |      |
|      |  top_tally_region (3) |      |
|______|_______________________|______| _ _ _ _ _ pitch/2
|      |                       |      |
|left  |         _____         |right |
|tally |        /     \        |tally |
|region|       /       \       |region|
| (2)  |      |    *    |      | (1)  | ___ > x
|      |       \  fuel /       |      |
|      |        \_____/        |      |
|      |       moderator       |      |
|______|_______________________|______| _ _ _ _ _ -pitch/2
|      |                       |      |
|      |bottom_tally_region (4)|      |
|______|_______________________|______| _ _ _ _ _ -3/4 pitch
       
|      |
    -pitch/2
|
-3/4 pitch

2. The labeling of the surfaces as 1,2,3,4 is assumed to be consistent with the tally ID
3. Spatial and angular meshes have uniform spacing (implicit in the chosen normalization)
4. All surfaces have the same number of anglular and spatial points (i.e. the same N_space, N_angle)
"""

import openmc
import pincell_moment_utils.config as config
from typing import List, Callable, Union
import numpy as np
from scipy.special import legendre
from scipy.integrate import simpson, quad
import itertools
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

pitch = config.PITCH
TRANSFORM_FUNCTIONS = config.TRANSFORM_FUNCTIONS
WEIGHT_FUNCTIONS = config.WEIGHT_FUNCTIONS
ANGULAR_BOUNDS = config.ANGULAR_BOUNDS
SPATIAL_BOUNDS = config.SPATIAL_BOUNDS

class SurfaceMeshTally:
    """A class for postprocessing surface flux mesh tallies"""
    meshes: List[List[np.ndarray]]
    """A list that contains a list of the spatial, angular, and energy  meshes for the surfaces in order according to the convention."""
    fluxes: List[np.ndarray]
    """A list of the processed and normalized array corresponding to the surface flux in the conventional surface order"""
    def __init__(self, statepoint_filename: str):
        self.statepoint = openmc.StatePoint(statepoint_filename)
        self.extract_surface_flux()

    def extract_surface_flux(self) -> None:
        # Store spatial meshes in a list for each surface
        self.meshes = []
        self.fluxes = []
        self.energy_filters = []
        for id in range(1, 5):
            tally = self.statepoint.get_tally(id=id)
            spatial_vals, angle_vals, energy_vals, energy_filter = self._extract_meshes(tally)

            # Number of space, angle, and energy points in mesh
            N_space, N_angle, N_energy = len(spatial_vals), len(angle_vals), len(energy_vals)

            self.meshes.append([spatial_vals, angle_vals, energy_vals])
            self.energy_filters.append(energy_filter)

            # Filter out high uncertainty mesh tallies
            flux = tally.mean  # The flux values (mean)
            flux_uncertainty = tally.std_dev  # The standard deviation
            flux[np.where(flux <= flux_uncertainty)] = 0

            # Transform linear flux tally to np.ndarray
            if id == 2:
                # Need to permute angle values so that they are in counterclockwise order
                flux = flux[:, 0, 0].reshape([N_space, N_angle + 1, N_energy], order='C') # Extra artifactual bin [-π/2, π/2]
                angle_filter = tally.find_filter(openmc.AzimuthalFilter)
                angle_bins = angle_filter.bins
                extra_index = np.where(angle_bins[:, 0] == -np.pi/2)[0]

                # Now remove this extra index
                angle_bins = np.delete(angle_bins, extra_index, axis=0)
                flux = np.delete(flux, extra_index, axis=1)

                # Now sort in counterclockwise order
                angle_bins[np.where(angle_bins < 0)] += 2*np.pi
                sorted_indices = np.argsort(angle_bins[:,0])
                flux = flux[:, sorted_indices, :]
            else:
                flux = flux[:, 0, 0].reshape([N_space, N_angle, N_energy], order='C')
                
            # Perform normalization to get cell-average flux
            Δspace = pitch/N_space
            Δangle = angle_vals[1] - angle_vals[0]
            Δenergy = np.diff(energy_filter.bins)[:, 0]
            flux /= (Δspace * Δangle * Δenergy[np.newaxis, np.newaxis, :])
            self.fluxes.append(flux)

    def _extract_meshes(self, tally) -> List[np.ndarray]:
        """Extract meshes from tallies that have them. Assumed spatial, angular, and energy meshes.
        
        Returns
        -------
            A list of numpy arrays corresponding to the spatial, angular, and energy meshes, as well as the energy filter used for normalization
        """

        spatial_filter = tally.find_filter(openmc.MeshFilter) # Only useful for getting N, doesn't actually give useful bins
        energy_filter = tally.find_filter(openmc.EnergyFilter)
        angle_filter = tally.find_filter(openmc.AzimuthalFilter)

        # Extract angular mesh
        if tally.id == 2: # Special treatment of angular branch cut on tally surface 2
            N_angle = len(angle_filter.bins) -1
            angle_bins = np.linspace(np.pi/2, 3/2*np.pi, N_angle + 1)
            angle_vals = (angle_bins[:-1] + angle_bins[1:]) / 2
        else:
            angle_vals = (angle_filter.values[:-1] + angle_filter.values[1:]) / 2

        # Extract spatial mesh (must be done manually using the known pitch and number of bins)
        N_space = len(spatial_filter.bins)
        bins = np.linspace(-pitch / 2, pitch / 2, N_space + 1)
        spatial_vals = (bins[:-1] + bins[1:]) / 2

        # Extract energy mesh
        energy_vals = (energy_filter.values[:-1] + energy_filter.values[1:]) / 2

        return [spatial_vals, angle_vals, energy_vals, energy_filter]


def compute_moments(mesh_tally: SurfaceMeshTally, I: int, J: int) -> np.ndarray:
    """Compute moments of a Fourier spatial expansion and a Legendre angular expansion for each surface in a given mesh tally object
    
    Parameters
    ----------
    mesh_tally
        The mesh `SurfaceMeshTally` object to use for computing the moments
    I
        Spatial expansion maximum index, 0 ≤ i ≤ I
    J
        Angular expansion maximum index, 0 ≤ j ≤ J"""
    
    _, _, energy_vals = mesh_tally.meshes[0]
    N_energy = len(energy_vals)

    coefs = np.zeros((4, I, J, N_energy, 2))  # Coefficients

    for surface in range(4):
        space_vals, angle_vals, _ = mesh_tally.meshes[surface]
        flux = mesh_tally.fluxes[surface]

        # Precompute basis functions
        basis_cache = _precompute_basis_functions(space_vals, angle_vals, I, J, surface, moment_calculation=True)

        for i, j, vector_index in itertools.product(range(I), range(J), range(2)):
            # Retrieve precomputed basis function
            integrand_eval = basis_cache[(i, j, vector_index)]

            for k in range(N_energy):
                # Compute product of flux and integrand
                product = flux[:, :, k] * integrand_eval

                # Perform integration
                if i == 0: # Integral over cosine with zero angular frequency causes different prefactor
                    if vector_index == 0:
                        coefs[surface, i, j, k, vector_index] = (2*j+1)/(2*pitch)*simpson(simpson(product, space_vals, axis=0), angle_vals, axis=0)
                    else: # Sine basis integral is zero
                        coefs[surface, i, j, k, vector_index] = 0
                else:
                    coefs[surface, i, j, k, vector_index] = (2*j+1)/(pitch)*simpson(simpson(product, space_vals, axis=0), angle_vals, axis=0)

    return coefs

def _precompute_basis_functions(space_vals: np.ndarray, angle_vals: np.ndarray, I: int, J: int, surface: int, 
                               moment_calculation: bool=False):
    """Function used for precomputing basis functions on a grid, for fast evaluation of moments and function expansions.
    
    Parameters
    ----------
    space_vals
        Spatial mesh to precompute basis functions on
    angle_vals
        Angular mesh to precompute basis functions on
    I
        Maximum spatial expansion index
    J
        Maximum angular expansion index
    surface
        Surface index (affects the form of the Legendre transformation functions and weights if used)
    moment_calculation
        Whether or not these basis functions will be used to compute moments. If so, also include the weighting function in for proper
        integration
    """
    basis_cache = {}
    transform_function = TRANSFORM_FUNCTIONS[surface]
    
    if moment_calculation:
        weight_function = WEIGHT_FUNCTIONS[surface]
    else:
        weight_function = lambda x: 1

    for i, j in itertools.product(range(I), range(J)):
        cos_basis = np.cos(i * np.pi * space_vals[:, None] / (pitch / 2))  # Precompute spatial part
        leg_basis = legendre(j)(transform_function(angle_vals)) * weight_function(angle_vals)  # Precompute angular part
        basis_cache[(i, j, 0)] = cos_basis * leg_basis
        
        sin_basis = np.sin(i * np.pi * space_vals[:, None] / (pitch / 2))
        basis_cache[(i, j, 1)] = sin_basis * leg_basis
    
    return basis_cache


class ReconstructedFlux:
    """Callable class to represent the reconstructed flux function for a given surface."""

    def __init__(self, coefficients, energy_filters, I, J, surface):
        """
        Initialize the reconstructed flux function.

        Parameters:
            coefficients : np.ndarray
                The coefficients of the functional expansion.
            energy_filters : list
                The energy filters for the given surface.
            I : int
                Spatial expansion max index.
            J : int
                Angular expansion max index.
            surface : int
                The surface index.
        """
        self.coefficients = coefficients
        self.energy_filters = energy_filters
        self.I = I
        self.J = J
        self.surface = surface

    def __call__(self, y, ω, E):
        """Evaluate the reconstructed flux at a given spatial (y), angular (ω), and energy (E) point."""
        # Determine the energy index for the given E
        E_idx = 0
        for bin_idx, bin in enumerate(self.energy_filters[self.surface].bins):
            if bin[0] <= E <= bin[1]:
                E_idx = bin_idx

        flux = 0
        # Loop through the basis function indices
        for i, j, vector_index in itertools.product(range(self.I), range(self.J), range(2)):
            flux += (
                self.coefficients[self.surface, i, j, E_idx, vector_index]
                * _basis_function(i, j, vector_index, self.surface)(y, ω)
            )

        # Return the flux value, ensuring it is non-negative
        return np.maximum(flux, 0)


class SurfaceExpansion:
    """Used for creating and evaluating the functional expansion of the surface flux from a given set of moments/coefficients"""
    energy_filters: list
    """Energy filters for the given expansion on each surface"""
    energy_bounds: List[List[float]]
    """The upper and lower energy bounds for the flux expansion on each surface"""
    I: int
    """Spatial expansion max index, 0 ≤ i ≤ I"""
    J: int
    """Angular expansion max index, 0 ≤ j ≤ J"""
    flux_functions: List[Callable[[float, float, float], float]]
    """List of surface flux functions for each of the surfaces"""
    coefficients: np.ndarray
    """Coefficients of the functional expansion of the flux on each surface, of shape (4 × I × J × N_energy × 2)"""
    def __init__(self, coefficients: np.ndarray, energy_filters: list):
        """Coefficients assumed to be of shape (4 × I × J × N_energy × 2) where I is the spatial expansion max index, J is the angular
        expansion max index, and N_energy is the number of energy groups."""
        
        _, self.I, self.J, self.N_energy, _ = coefficients.shape
        self.coefficients = coefficients
        self.energy_filters = energy_filters
        self.energy_bounds = self._get_energy_bounds()
        self.flux_functions = [self._construct_surface_expansion(surface) for surface in range(4) ]

    def normalize_by(self, normalization_const: Union[float, List[float]]) -> None:
        """Normalize all surface flux functions by a global normalization constant, or normalize each surface by an individual normalization
        constant
        
        Parameters
        ----------
        normalization_const
            Either a single global normalization constant for all surfaces, or a list of normalization constants for each surface"""
        if isinstance(normalization_const, float):
            self.coefficients = self.coefficients/normalization_const
        elif isinstance(normalization_const, list) or isinstance(normalization_const, np.ndarray):
            for surface in range(4):
                self.coefficients[surface, :, :, :, :] /= normalization_const[surface]
        else:
            raise ValueError(f"Normalization constant can only be of type float or list, you supplied type {type(normalization_const)}")
        
        # Now reconstruct the flux functions with these normalized coefficients
        self.flux_functions = [self._construct_surface_expansion(surface) for surface in range(4) ]

    def integrate_flux(self, surface: int) -> float:
        """Compute the integral of the surface flux (over the relevant surface phase space)."""
        spatial_bounds = SPATIAL_BOUNDS[surface]
        angular_bounds = ANGULAR_BOUNDS[surface]

        energy_filter = self.energy_filters[surface]
        Δenergy = np.diff(energy_filter.bins)[:, 0]
        energy_vals = (energy_filter.values[:-1] + energy_filter.values[1:]) / 2
        
        # To perform integration over energy values, we just need to choose one energy point from each bin, compute the spatial-angular
        # integral, then multiply by the energy bin width
        integral = 0
        for energy_index in range(len(energy_vals)):
            # Reconstructed integrated functional expansion
            energy_integral = 0
            for i, j, vector_index in itertools.product(range(self.I), range(self.J), range(2)):
                energy_integral += self.coefficients[surface, i, j, energy_index, vector_index] * _integral_basis_function(i, j, vector_index, surface)(
                    spatial_bounds[0], spatial_bounds[1], angular_bounds[0], angular_bounds[1])
            
            integral += Δenergy[energy_index]*np.max(energy_integral, 0) # To eliminate nonphysical negative fluxes

        return integral
    
    def generate_samples(self, N: int, num_cores: int = multiprocessing.cpu_count()):
        """Generate N samples from the flux functions across all surfaces in accordance with their relative norms."""
        samples = []

        # Compute the normalization factors for the PDF over all surfaces
        norm_consts = np.zeros(4)
        for surface in range(4):
            norm_consts[surface] = self.integrate_flux(surface)

        # Normalize each of the individual surface flux functions by their respective norms
        self.normalize_by(norm_consts)

        # Compute the number of samples to generate from each surface
        N_surface = np.floor(N * norm_consts / np.sum(norm_consts)).astype(int)
        N_surface[3] += N - np.sum(N_surface)  # Ensure the total number of samples is N

        for surface in range(4):
            spatial_bounds = config.SPATIAL_BOUNDS
            angular_bounds = config.ANGULAR_BOUNDS
            energy_bounds = self.energy_bounds
            domain = [
                (bounds[0], bounds[1])
                for bounds in [spatial_bounds[surface], angular_bounds[surface], energy_bounds[surface]]
            ]
            samples.append(
                self.rejection_sampling_3d_parallel(
                    surface,
                    domain,
                    N_surface[surface],
                    num_cores
                )
            )

        # Undo the surface normalization
        self.normalize_by(1 / norm_consts)

        return samples

    def _get_energy_bounds(self) -> List[List[float]]:
        """Returns a list of energy bounds for each surface"""
        energy_bounds = [ [ self.energy_filters[surface].bins[0,0], self.energy_filters[surface].bins[-1,1] ] for surface in range(4) ]
        return energy_bounds

    def _construct_surface_expansion(self, surface: int) -> ReconstructedFlux:
        """Construct the surface expansion of the flux from the basis functions."""
        return ReconstructedFlux(
            coefficients=self.coefficients,
            energy_filters=self.energy_filters,
            I=self.I,
            J=self.J,
            surface=surface
        )

    def evaluate_on_grid(self, surface: int, grid_points: np.ndarray):
        """Evaluate the flux on a grid of spatial, angular, and energy points"""
        space_vals, angle_vals, energy_vals = grid_points
        flux = np.zeros((len(space_vals), len(angle_vals), len(energy_vals)))

        # Precompute basis functions
        basis_cache = _precompute_basis_functions(space_vals, angle_vals, self.I, self.J, surface)

        for i, j, vector_index in itertools.product(range(self.I), range(self.J), range(2)):
            for k, E in enumerate(energy_vals):
                E_idx = 0
                for bin_idx, bin in enumerate(self.energy_filters[surface].bins):
                    if bin[0] <= E <= bin[1]:
                        E_idx = bin_idx
                coef = self.coefficients[surface, i, j, E_idx, vector_index]
                flux[:, :, k] += coef * basis_cache[(i, j, vector_index)]

        return np.maximum(flux, 0) # To avoid returning nonphysical negative values
    

    def rejection_sampling_3d_parallel(
        self,
        surface: int,
        domain: List[List[float]],
        num_samples: int,
        num_workers: int = None
    ) -> np.ndarray:
        """
        Perform parallel rejection sampling for a 3-variable probability distribution.
        """
        # Extract bounds from the domain
        x_bounds, y_bounds, z_bounds = domain
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds
        z_min, z_max = z_bounds

        # Compute the volume of the domain
        domain_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        proposal_pdf_value = 1 / domain_volume

        # Estimate the maximum value of the target PDF
        # You might need to adjust this depending on your specific PDF
        # Could use a grid search, optimization, or analytical maximum if known
        M = self.estimate_max_value(surface, domain)

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        samples_per_worker = [num_samples // num_workers] * num_workers
        for i in range(num_samples % num_workers):
            samples_per_worker[i] += 1

        target_pdf = self.flux_functions[surface]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    rejection_sampling_worker,
                    x_bounds,
                    y_bounds,
                    z_bounds,
                    proposal_pdf_value,
                    n,
                    target_pdf,
                    M,
                )
                for n in samples_per_worker
            ]

            results = [future.result() for future in futures]

        all_samples = [sample for result in results for sample in result]
        return np.array(all_samples)

    def estimate_max_value(self, surface, domain: List[List[float]], 
                        grid_points: int = 20) -> float:
        """
        Estimate the maximum value of the PDF over its domain using a grid search.
        A more sophisticated method might be needed depending on the PDF.
        """
        x_bounds, y_bounds, z_bounds = domain
        x = np.linspace(x_bounds[0], x_bounds[1], grid_points)
        y = np.linspace(y_bounds[0], y_bounds[1], grid_points)
        z = np.linspace(z_bounds[0], z_bounds[1], grid_points)
        
        pdf_vals = self.evaluate_on_grid(surface, [x, y, z])
        max_val = np.max(pdf_vals)
        
        # Add safety factor to ensure we don't miss the true maximum
        return max_val * 1.1  # 10% safety margin


def _basis_function(i: int, j: int, vector_index: int, surface: int) -> Callable[[float, float], float]:
    """Gets the vector_index index of the ith spatial and jth angular basis function on the given surface"""
    transform_function = TRANSFORM_FUNCTIONS[surface]
    spatial_bounds = SPATIAL_BOUNDS[surface]
    angular_bounds = ANGULAR_BOUNDS[surface]
    if vector_index == 0:
        def basis(x, ω):
            if not ( spatial_bounds[0] <= x <= spatial_bounds[1] ):
                raise ValueError(f"Supplied spatial point {x} is not within the spatial range [{spatial_bounds[0]}, {spatial_bounds[1]}]")
            if not ( angular_bounds[0] <= ω <= angular_bounds[1] ):
                raise ValueError(f"Supplied angle {ω} is not within the angular range [{angular_bounds[0]}, {angular_bounds[1]}]")
            return np.cos(i*np.pi* x/(pitch/2))*legendre(j)(transform_function(ω))
    elif vector_index == 1:
        def basis(x, ω):
            if not ( spatial_bounds[0] <= x <= spatial_bounds[1] ):
                raise ValueError(f"Supplied spatial point {x} is not within the spatial range [{spatial_bounds[0]}, {spatial_bounds[1]}]")
            if not ( angular_bounds[0] <= ω <= angular_bounds[1] ):
                raise ValueError(f"Supplied angle {ω} is not within the angular range [{angular_bounds[0]}, {angular_bounds[1]}]")
            return np.sin(i*np.pi* x/(pitch/2))*legendre(j)(transform_function(ω))
    else:
        raise ValueError(f"vector_index must be 0 or 1, you supplied {vector_index}")
    return basis

def _integral_basis_function(i: int, j: int, vector_index: int, surface: int) -> Callable[[float, float, float, float], float]:
    """Gets the vector_index index of the ith spatial and jth angular integrated basis function on the given surface. Note that
    the Fourier basis functions may be analytically integrated, and by comparison with the above functions for the basis functions
    you can verify the simple integration rule used. Due to the transform function, however, to integrate the Legendre basis function we need
    to use a quadrature rule. Note also the `i != 0` statements are to prevent division by zero."""
    transform_function = TRANSFORM_FUNCTIONS[surface]
    spatial_bounds = SPATIAL_BOUNDS[surface]
    angular_bounds = ANGULAR_BOUNDS[surface]
    if vector_index == 0:
        def basis(x_lower, x_upper, ω_lower, ω_upper):
            for x, ω in zip([x_lower, x_upper], [ω_lower, ω_upper]):
                if not ( spatial_bounds[0] <= x <= spatial_bounds[1] ):
                    raise ValueError(f"Supplied spatial point {x} is not within the spatial range [{spatial_bounds[0]}, {spatial_bounds[1]}]")
                if not ( angular_bounds[0] <= ω <= angular_bounds[1] ):
                    raise ValueError(f"Supplied angle {ω} is not within the angular range [{angular_bounds[0]}, {angular_bounds[1]}]")
            if i != 0:
                return ((pitch/2)/(i*np.pi)* np.sin(i*np.pi* x_upper/(pitch/2)) - 
                        (pitch/2)/(i*np.pi)* np.sin(i*np.pi* x_lower/(pitch/2)) ) * quad(lambda ω: legendre(j)(transform_function(ω)), ω_lower, ω_upper)[0]
            else:
                return (x_upper-x_lower) * quad(lambda ω: legendre(j)(transform_function(ω)), ω_lower, ω_upper)[0]
    elif vector_index == 1:
        def basis(x_lower, x_upper, ω_lower, ω_upper): # Returns the integral basis fuction integrated over 
            for x, ω in zip([x_lower, x_upper], [ω_lower, ω_upper]):
                if not ( spatial_bounds[0] <= x <= spatial_bounds[1] ):
                    raise ValueError(f"Supplied spatial point {x} is not within the spatial range [{spatial_bounds[0]}, {spatial_bounds[1]}]")
                if not ( angular_bounds[0] <= ω <= angular_bounds[1] ):
                    raise ValueError(f"Supplied angle {ω} is not within the angular range [{angular_bounds[0]}, {angular_bounds[1]}]")
            if i != 0:
                return (-(pitch/2)/(i*np.pi)* np.cos(i*np.pi* x_upper/(pitch/2)) + 
                        (pitch/2)/(i*np.pi)* np.cos(i*np.pi* x_lower/(pitch/2)) )*quad(lambda ω: legendre(j)(transform_function(ω)), ω_lower, ω_upper)[0]
            else:
                return 0
    else:
        raise ValueError(f"vector_index must be 0 or 1, you supplied {vector_index}")
    return basis


def rejection_sampling_worker(
    x_bounds: List[float],
    y_bounds: List[float],
    z_bounds: List[float],
    proposal_pdf_value: float,
    num_samples: int,
    target_pdf: Callable[[float, float, float], float],
    M: float,  # Added scaling factor
):
    """
    Worker function for generating samples using rejection sampling.
    
    Parameters:
        x_bounds: The bounds for the x-dimension.
        y_bounds: The bounds for the y-dimension.
        z_bounds: The bounds for the z-dimension.
        proposal_pdf_value: The proposal PDF value (uniform over the domain).
        num_samples: The number of samples to generate.
        target_pdf: The target PDF function.
        M: Scaling factor ensuring M*g(x) ≥ f(x)
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    z_min, z_max = z_bounds

    local_samples = []
    while len(local_samples) < num_samples:
        # Sample a candidate point from the uniform proposal
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        z = np.random.uniform(z_min, z_max)
        candidate = np.array([x, y, z])

        # Generate a uniform random number in [0, 1]
        u = np.random.uniform(0, 1)

        # Correct acceptance criterion
        if u < target_pdf(*candidate) / (M * proposal_pdf_value):
            local_samples.append(candidate)

    return local_samples