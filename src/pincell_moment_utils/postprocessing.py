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
import multiprocessing
from pincell_moment_utils.sampling import sample_surface_flux
from abc import ABC, abstractmethod

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


class SurfaceExpansionBase(ABC):
    """Used for creating and evaluating the functional expansion of the surface flux from a given set of moments/coefficients"""
    energy_filters: list
    """Energy filters for the given expansion on each surface"""
    energy_bounds: List[List[float]]
    """The upper and lower energy bounds for the flux expansion on each surface"""
    I: int
    """Spatial expansion order (number of terms in spatial expansion)"""
    J: int
    """Angular expansion order (number of terms in angular expansion)"""
    flux_functions: List[Callable[[float, float, float], float]]
    """List of surface flux functions for each of the surfaces"""
    coefficients: np.ndarray
    """Coefficients of the functional expansion of the flux on each surface, of shape (4 × I × J × N_energy × 2) for Fourier-Legendre, or
    (4 × I × J × N_energy) for Bernstein-Bernstein"""
    def __init__(self, coefficients: np.ndarray, energy_filters: list):
        """
        Parameters
        ----------
        coefficients : np.ndarray
            Coefficients of shape (4, I, J, N_energy, 2), or some shape
            that is consistent with the type of expansion.
        energy_filters : list
            List of energy filters for the 4 surfaces.
        """
        # Common attributes that all expansions share:
        # shape = (4, I, J, N_energy, 2) for example
        _, self.I, self.J, self.N_energy, _ = coefficients.shape
        self.coefficients = coefficients
        self.energy_filters = energy_filters
        
        # Just storing these for convenience if your code references them
        self.flux_functions = [None]*4 
        self.energy_bounds = self._get_energy_bounds()

    @abstractmethod
    def _basis_function(self, i: int, j: int, vector_index: int, surface: int) -> Callable:
        """
        Return the callable that represents the i,j,(cos/sin) basis function at a
        point (x, ω). Must be overridden by subclasses.
        """
        pass

    @abstractmethod
    def _precompute_basis_functions(self, space_vals: np.ndarray, angle_vals: np.ndarray,
                                    surface: int):
        """
        Return a cache (dict or similar) of basis-function values evaluated on
        the grids space_vals × angle_vals. Subclasses decide exactly how.
        """
        pass

    @abstractmethod
    def _integral_basis_function(self, i: int, j: int, vector_index: int, surface: int) -> Callable:
        """
        Return a function that, given x-limits and ω-limits, yields the integral
        of the i,j basis over that domain. Subclasses implement actual logic.
        """
        pass

    @abstractmethod
    def _construct_surface_expansion(self, surface: int) -> Callable:
        """
        Build a single-surface 'ReconstructedFlux' or similar object that can
        evaluate the flux at (x, ω, E). Subclass decides details.
        """
        pass

    def _get_energy_bounds(self):
        """Returns a list of energy bounds for each surface."""
        return [
            [f.bins[0, 0], f.bins[-1, 1]]
            for f in self.energy_filters
        ]


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

    def normalize_by(self, normalization_const: Union[float, List[float]]) -> None:
        """Normalize all surface flux functions by a global normalization constant, or normalize each surface by an individual normalization
        constant
        
        Parameters
        ----------
        normalization_const
            Either a single global normalization constant for all surfaces, or a list of normalization constants for each surface"""
        if isinstance(normalization_const, float):
            self.coefficients = self.coefficients / normalization_const
        elif isinstance(normalization_const, (list, np.ndarray)):
            for surface in range(4):
                self.coefficients[surface, :, :, :, :] /= normalization_const[surface]
        else:
            raise ValueError(f"Unsupported normalization_const type: {type(normalization_const)}")

        # Re-build the flux functions after normalizing
        self.flux_functions = [
            self._construct_surface_expansion(sfc)
            for sfc in range(4)
        ]

    def evaluate_on_grid(self, surface: int, grid_points: np.ndarray):
        """
        Evaluate the flux on a 3D grid: (space_vals, angle_vals, energy_vals).
        """
        space_vals, angle_vals, energy_vals = grid_points
        flux = np.zeros((len(space_vals), len(angle_vals), len(energy_vals)))

        basis_cache = self._precompute_basis_functions(space_vals, angle_vals, surface)

        # For each (i, j, vector_index), multiply by coefficient
        for i, j, vector_index in itertools.product(range(self.I), range(self.J), range(2)):
            for k, E in enumerate(energy_vals):
                # Figure out which energy bin E falls into
                e_idx = 0
                for bin_idx, energy_bin in enumerate(self.energy_filters[surface].bins):
                    if energy_bin[0] <= E <= energy_bin[1]:
                        e_idx = bin_idx
                        break
                coef = self.coefficients[surface, i, j, e_idx, vector_index]
                flux[:, :, k] += coef * basis_cache[(i, j, vector_index)]

        # Optionally clamp to zero
        return np.maximum(flux, 0)

    def generate_samples(
        self, N: int, sample_surface=None, num_cores: int = multiprocessing.cpu_count(),
        method: str = 'ensemble', use_log_energy: bool = True, burn_in: int = 1000,
        progress=False
    ):
        """
        High-level API for drawing samples from the flux expansions on each surface.
        
        Parameters
        ----------
        N
            total number of samples across all surfaces
        surface
            default behavior is to generate samples for all surfaces, but if a surface index is specified, samples will only be generated
            for the given surface
        num_cores
            number of parallel cores to use (for some sampling methods)
        method
            which sampler to use ('rejection', 'ensemble', 'metropolis_hastings')
        use_log_energy
            sample in log(E)-space if True
        burn_in
            burn-in steps for MCMC methods
        progres
            whether or not to print progress messages when doing ensemble sampling
        """
        if sample_surface is not None:
            surfaces = [sample_surface]
        else:
            surfaces = [0, 1, 2, 3]

        # 1) Compute integrals to get normalizations
        norm_consts = np.ones(4)
        for sfc in surfaces:
            norm_consts[sfc] = self.integrate_flux(sfc)

        # 2) Normalize
        self.normalize_by(norm_consts)

        # 3) Determine how many samples per surface
        N_surface = np.floor(N * norm_consts / np.sum(norm_consts)).astype(int)
        N_surface[-1] += N - np.sum(N_surface)

        # 4) Generate samples
        all_samples = [[] for _ in range(4)]
        for sfc in surfaces:
            # Build domain from config
            sp_bounds = config.SPATIAL_BOUNDS[sfc]
            w_bounds = config.ANGULAR_BOUNDS[sfc]
            e_bounds = self.energy_bounds[sfc]

            if use_log_energy:
                domain = [
                    sp_bounds,
                    w_bounds,
                    (np.log(e_bounds[0]), np.log(e_bounds[1]))
                ]
            else:
                domain = [
                    sp_bounds,
                    w_bounds,
                    (e_bounds[0], e_bounds[1])
                ]

            n_samps = int(N_surface[sfc]) if sample_surface is None else N
            sampler_pdf = self.flux_functions[sfc]  # from _construct_surface_expansion

            samples_sfc = sample_surface_flux(
                pdf=sampler_pdf,
                domain=domain,
                N=n_samps,
                method=method,
                use_log_energy=use_log_energy,
                burn_in=burn_in,
                num_cores=num_cores,
                progress=progress
            )
            all_samples[sfc] = samples_sfc

        # 5) Undo PDF normalization to restore original scale
        self.normalize_by(1.0 / norm_consts)

        return all_samples

    def estimate_max_point(self, surface, grid_points: int = 20):
        """
        Estimate maximum flux location for the given surface.
        """
        space_bounds = config.SPATIAL_BOUNDS[surface]
        angle_bounds = config.ANGULAR_BOUNDS[surface]
        e_bounds = self.energy_bounds[surface]

        space_vals = np.linspace(space_bounds[0], space_bounds[1], grid_points)
        angle_vals = np.linspace(angle_bounds[0], angle_bounds[1], grid_points)
        energy_vals = np.linspace(e_bounds[0], e_bounds[1], grid_points)

        pdf_vals = self.evaluate_on_grid(surface, [space_vals, angle_vals, energy_vals])
        max_idx = np.argmax(pdf_vals)
        max_idx = np.unravel_index(max_idx, pdf_vals.shape)
        return (space_vals[max_idx[0]], angle_vals[max_idx[1]], energy_vals[max_idx[2]])


class FourierLegendreExpansion(SurfaceExpansionBase):
    """
    A concrete implementation of the Fourier (cos/sin) in space
    and Legendre polynomial in angle expansions.
    """
    def _basis_function(self, i: int, j: int, vector_index: int, surface: int):
        """
        Return a function f(x, ω) = [cos(...)*P_j(...) or sin(...)*P_j(...)].
        """
        transform_function = TRANSFORM_FUNCTIONS[surface]
        sbounds = SPATIAL_BOUNDS[surface]
        abounds = ANGULAR_BOUNDS[surface]

        # vector_index=0 => cos-basis, =1 => sin-basis
        if vector_index == 0:
            def basis_fn(x, ω):
                # Optional boundary checks
                if not (sbounds[0] <= x <= sbounds[1]):
                    raise ValueError(f"x={x} not in {sbounds}")
                if not (abounds[0] <= ω <= abounds[1]):
                    raise ValueError(f"ω={ω} not in {abounds}")
                return np.cos(i * np.pi * x/(pitch/2)) * legendre(j)(transform_function(ω))
        else:
            def basis_fn(x, ω):
                if not (sbounds[0] <= x <= sbounds[1]):
                    raise ValueError(f"x={x} not in {sbounds}")
                if not (abounds[0] <= ω <= abounds[1]):
                    raise ValueError(f"ω={ω} not in {abounds}")
                return np.sin(i * np.pi * x/(pitch/2)) * legendre(j)(transform_function(ω))

        return basis_fn

    def _precompute_basis_functions(self, space_vals: np.ndarray, angle_vals: np.ndarray, surface: int):
        """
        Build a dictionary for all (i, j, vector_index) evaluated on space_vals×angle_vals.
        """
        transform_function = TRANSFORM_FUNCTIONS[surface]

        basis_cache = {}
        for i, j, vector_index in itertools.product(range(self.I), range(self.J), range(2)):
            if vector_index == 0:
                cos_basis = np.cos(i*np.pi* space_vals[:, None]/(pitch/2))
                leg_basis = legendre(j)(transform_function(angle_vals))
                basis_cache[(i, j, 0)] = cos_basis * leg_basis
            else:
                sin_basis = np.sin(i*np.pi* space_vals[:, None]/(pitch/2))
                leg_basis = legendre(j)(transform_function(angle_vals))
                basis_cache[(i, j, 1)] = sin_basis * leg_basis
        return basis_cache

    def _integral_basis_function(self, i: int, j: int, vector_index: int, surface: int):
        """
        Return the integral over x in [x_lower, x_upper] and ω in [ω_lower, ω_upper].
        """
        transform_function = TRANSFORM_FUNCTIONS[surface]

        def integrand_angle(ω):
            return legendre(j)(transform_function(ω))

        if vector_index == 0:
            # cos part
            def f_int(x_lower, x_upper, w_lower, w_upper):
                # integrate cos in x
                if i == 0:
                    # integral of cos(0 * x) dx = x
                    x_part = x_upper - x_lower
                else:
                    # integral cos(k x) dx = (1/k) sin(k x)
                    k = i*np.pi/(pitch/2)
                    x_part = (1.0/k)*( np.sin(k*x_upper) - np.sin(k*x_lower) )
                
                # integrate Legendre in ω (needs numeric quadrature, or possibly known antiderivative for Legendre)
                w_part, _ = quad(integrand_angle, w_lower, w_upper)
                
                return x_part * w_part

        else:
            # sin part
            def f_int(x_lower, x_upper, w_lower, w_upper):
                if i == 0:
                    # integral sin(0 * x) dx = 0
                    return 0.0
                else:
                    k = i*np.pi/(pitch/2)
                    # integral sin(k x) dx = (1/k)(-cos(k x))
                    x_part = (1.0/k)*(-np.cos(k*x_upper) + np.cos(k*x_lower))
                
                w_part, _ = quad(integrand_angle, w_lower, w_upper)
                return x_part * w_part

        return f_int

    def _construct_surface_expansion(self, surface: int) -> Callable:
        """
        Return a callable that, given (y, ω, E), returns flux.
        You can replicate your original ReconstructedFlux pattern or inline it.
        """
        # If you want to replicate your ReconstructedFlux class, you can do that here:
        return _FourierLegendreReconstructedFlux(
            coefficients=self.coefficients,
            energy_filters=self.energy_filters,
            I=self.I,
            J=self.J,
            surface=surface
        )


class _FourierLegendreReconstructedFlux:
    """
    Minimal wrapper for the flux function on one surface, similar to your ReconstructedFlux.
    """
    def __init__(self, coefficients, energy_filters, I, J, surface):
        self.coefficients = coefficients
        self.energy_filters = energy_filters
        self.I = I
        self.J = J
        self.surface = surface

    def __call__(self, y, ω, E):
        # figure out E bin
        e_idx = 0
        for bin_idx, bin_pair in enumerate(self.energy_filters[self.surface].bins):
            if bin_pair[0] <= E <= bin_pair[1]:
                e_idx = bin_idx
                break

        # sum expansions
        transform_fn = TRANSFORM_FUNCTIONS[self.surface]
        val = 0.0
        for i, j, vector_index in itertools.product(range(self.I), range(self.J), range(2)):
            coef = self.coefficients[self.surface, i, j, e_idx, vector_index]
            angle_part = legendre(j)(transform_fn(ω))
            if vector_index == 0:
                # cos
                val += coef * np.cos(i*np.pi*y/(pitch/2)) * angle_part
            else:
                # sin
                val += coef * np.sin(i*np.pi*y/(pitch/2)) * angle_part

        return max(val, 0.0)


def surface_expansion(
    coefficients: np.ndarray,
    energy_filters: list,
    expansion_type: str = 'fourier_legendre'
) -> SurfaceExpansionBase:
    """
    Create the appropriate SurfaceExpansion subclass based on expansion_type.
    """
    if expansion_type.lower() == 'fourier_legendre':
        return FourierLegendreExpansion(coefficients, energy_filters)
    elif expansion_type.lower() == 'bernstein_bernstein':
        raise NotImplementedError("Bernstein bernstein expansion not yet implemented.")
    else:
        raise ValueError(f"Unknown expansion_type: {expansion_type}")
