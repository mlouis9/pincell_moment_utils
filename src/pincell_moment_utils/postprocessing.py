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
import emcee

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

def _global_log_prob(theta, pdf, domain):
    """
    A top-level log-prob function that can be pickled for multiprocessing.
    
    Parameters
    ----------
    theta : array_like
        A 3-element array [x, y, E].
    pdf : callable
        The flux function for a given surface, e.g. ReconstructedFlux.
    domain : list of 3 tuples
        [(x_min, x_max), (y_min, y_max), (E_min, E_max)].
        
    Returns
    -------
    float
        The log of the PDF (log(flux)), or -np.inf if out-of-bounds or flux<=0.
    """
    x, y, E = theta
    (x_min, x_max), (y_min, y_max), (e_min, e_max) = domain

    # Domain checks
    if not (x_min <= x <= x_max):
        return -np.inf
    if not (y_min <= y <= y_max):
        return -np.inf
    if not (e_min <= E <= e_max):
        return -np.inf
    
    val = pdf(x, y, E)
    if val <= 0:
        return -np.inf
    return np.log(val)

def _global_log_prob_logE(theta, pdf, domain):
    """
    theta: [x, w, lnE]
        The Markov chain parameters in spatial, angular, and log-energy space.
    pdf: a ReconstructedFlux-like object, e.g. self.flux_functions[surface]
    domain: [(x_min, x_max), (w_min, w_max), (lnE_min, lnE_max)]
    """
    x, w, lnE = theta
    (x_min, x_max), (w_min, w_max), (lnE_min, lnE_max) = domain

    # -- Check domain --
    if not (x_min <= x <= x_max):
        return -np.inf
    if not (w_min <= w <= w_max):
        return -np.inf
    if not (lnE_min <= lnE <= lnE_max):
        return -np.inf

    # Convert lnE -> E
    E = np.exp(lnE)

    # Evaluate the flux at (x, w, E)
    val = pdf(x, w, E)
    # Multiply by the Jacobian factor (E) for logE coords
    val *= E

    # If flux <= 0, log is -inf
    if val <= 0:
        return -np.inf
    return np.log(val)


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
            self.coefficients = self.coefficients/normalization_const
        elif isinstance(normalization_const, list) or isinstance(normalization_const, np.ndarray):
            for surface in range(4):
                self.coefficients[surface, :, :, :, :] /= normalization_const[surface]
        else:
            raise ValueError(f"Normalization constant can only be of type float or list, you supplied type {type(normalization_const)}")
        
        # Now reconstruct the flux functions with these normalized coefficients
        self.flux_functions = [self._construct_surface_expansion(surface) for surface in range(4) ]

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
    
    def generate_samples(
        self,
        N: int,
        num_cores: int = multiprocessing.cpu_count(),
        method: str = 'metropolis_hastings',
        use_log_energy: bool = False
    ):
        """
        Generate N samples from the flux functions across all surfaces.

        Parameters
        ----------
        N : int
            The number of samples to generate
        num_cores : int
            The number of cores for parallel usage
        method : str
            'rejection', 'ensemble', or 'metropolis_hastings'
        use_log_energy : bool
            If True, sample in log(E)-space with Jacobian. If False, sample in E-space.
        """
        samples = []

        # 1) Compute normalization factors
        norm_consts = np.zeros(4)
        for sfc in range(4):
            norm_consts[sfc] = self.integrate_flux(sfc)

        # 2) Normalize the flux to act like a PDF
        self.normalize_by(norm_consts)

        # 3) Split the total N by surface proportionally
        N_surface = np.floor(N * norm_consts / np.sum(norm_consts)).astype(int)
        # Adjust last entry so total sums to N
        N_surface[-1] += N - np.sum(N_surface)

        # 4) For each surface, call the chosen method
        for surface in range(4):
            # Build domain
            sp_bounds = config.SPATIAL_BOUNDS[surface]      # (x_min, x_max)
            w_bounds  = config.ANGULAR_BOUNDS[surface]      # (w_min, w_max)
            E_bounds  = self.energy_bounds[surface]         # (E_min, E_max)

            if not use_log_energy:
                # Domain in linear E
                domain = [sp_bounds, w_bounds, (E_bounds[0], E_bounds[1])]
            else:
                # Domain in log(E)
                lnE_min = np.log(E_bounds[0])
                lnE_max = np.log(E_bounds[1])
                domain = [sp_bounds, w_bounds, (lnE_min, lnE_max)]

            n_samps = N_surface[surface]

            if method == 'rejection':
                if use_log_energy:
                    raise NotImplementedError("No log-E rejection sampler here. Use linear or write a custom one.")
                # existing linear rejection sampler
                out = self.rejection_sampling_3d_parallel(surface, domain, n_samps, num_cores)
                samples.append(out)

            elif method == 'ensemble':
                if not use_log_energy:
                    # old ensemble in linear space
                    out = self.ensemble(surface, domain, n_samps, num_cores=num_cores)
                else:
                    # new ensemble in logE
                    out = self.ensemble_logE(surface, domain, n_samps, num_cores=num_cores)
                samples.append(out)

            elif method == 'metropolis_hastings':
                if not use_log_energy:
                    # old MH in linear space
                    out = self.metropolis_hastings(surface, domain, n_samps)
                else:
                    # new MH in logE
                    out = self.metropolis_hastings_logE(surface, domain, n_samps)
                samples.append(out)

            else:
                raise ValueError("Unsupported sampling method.")

        # 5) Undo the flux normalization
        self.normalize_by(1 / norm_consts)

        return samples
    
    def ensemble(
        self,
        surface: int,
        domain,
        N: int,
        nwalkers: int = 32,
        burn_in: int = 1000,
        num_cores: int = 1
    ):
        """
        Ensemble sampler in (x, w, E) space returning exactly N total samples.
        """

        pdf = self.flux_functions[surface]

        # Initialize
        p0 = []
        for _ in range(nwalkers):
            x0 = np.random.uniform(domain[0][0], domain[0][1])
            w0 = np.random.uniform(domain[1][0], domain[1][1])
            e0 = np.random.uniform(domain[2][0], domain[2][1])
            p0.append([x0, w0, e0])
        p0 = np.array(p0)

        pool = None
        if num_cores is not None and num_cores > 1:
            pool = multiprocessing.Pool(processes=num_cores)

        try:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                3,
                _global_log_prob,  # you'd define at module scope
                args=(pdf, domain),
                pool=pool
            )
            # Burn-in
            state = sampler.run_mcmc(p0, burn_in, progress=True)
            sampler.reset()

            # Steps so that total ~ N
            nsteps = (N + nwalkers - 1) // nwalkers
            state = sampler.run_mcmc(state, nsteps, progress=True)

        finally:
            if pool is not None:
                pool.close()
                pool.join()

        chain = sampler.get_chain(discard=0, thin=1, flat=True)
        if chain.shape[0] > N:
            chain = chain[:N, :]
        return chain

    def ensemble_logE(
        self,
        surface: int,
        domain,
        N: int,
        nwalkers: int = 32,
        burn_in: int = 1000,
        num_cores: int = 1
    ):
        """
        Ensemble sampler in (x, w, lnE) space, returning exactly N total samples.

        Parameters
        ----------
        surface : int
            Surface index to sample from
        domain : list of 3 tuples [(x_min, x_max), (w_min, w_max), (lnE_min, lnE_max)]
            The domain in spatial, angular, and ln-energy space.
        N : int
            Total number of POST–burn-in samples to return.
        nwalkers : int
            Number of walkers in the ensemble.
        burn_in : int
            Number of burn-in steps (discarded).
        num_cores : int
            Number of CPU cores for parallel sampling. If <= 1, no parallelization.

        Returns
        -------
        samples : (N, 3) np.ndarray
            Flattened array of exactly N samples in real space (x, w, E).
            Each row is [x, w, E].
        """
        # ReconstructedFlux for this surface
        pdf = self.flux_functions[surface]

        # Initialize each walker's position in (x, w, lnE)
        p0 = []
        for _ in range(nwalkers):
            x0 = np.random.uniform(domain[0][0], domain[0][1])
            w0 = np.random.uniform(domain[1][0], domain[1][1])
            lnE0 = np.random.uniform(domain[2][0], domain[2][1])
            p0.append([x0, w0, lnE0])
        p0 = np.array(p0)

        # Create a pool if needed
        pool = None
        if num_cores is not None and num_cores > 1:
            pool = multiprocessing.Pool(processes=num_cores)

        try:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                3,
                _global_log_prob_logE,
                args=(pdf, domain),
                pool=pool
            )

            # ----- Burn-in -----
            state = sampler.run_mcmc(p0, burn_in, progress=True)
            sampler.reset()

            # We want exactly N total samples overall => each walker does nsteps
            # so that nwalkers*nsteps >= N
            nsteps = (N + nwalkers - 1) // nwalkers  # ceiling of (N / nwalkers)

            # ----- Production run -----
            state = sampler.run_mcmc(state, nsteps, progress=True)

        finally:
            if pool is not None:
                pool.close()
                pool.join()

        # Flatten chain => shape is (nwalkers*nsteps, 3)
        chain = sampler.get_chain(discard=0, thin=1, flat=True)

        # If we overshot, take first N
        if chain.shape[0] > N:
            chain = chain[:N, :]

        # Convert lnE -> E
        # "chain" is [ [x, w, lnE], ... ]
        out = []
        for (xx, ww, lnE) in chain:
            out.append([xx, ww, np.exp(lnE)])
        return np.array(out)

    
    def metropolis_hastings(self, surface: int, domain, N, x0=None, proposal_std=None, max_init_tries=500):
        """
        Perform Metropolis-Hastings sampling while respecting domain constraints.
        
        Parameters:
        -----------
        surface : int
            The surface whose flux function is being sampled
        domain : list of tuples [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
            Bounds for each variable in the order (spatial, angle, energy).
        N : int
            Number of samples to generate.
        x0 : np.ndarray, optional
            Initial point. If None, will be randomly chosen within the domain.
        proposal_std : np.ndarray, optional
            Standard deviation of the Gaussian proposal distribution for each variable.
            
        Returns:
        --------
        np.ndarray
            Array of shape (N, 3) for the accepted samples (the chain).
        """
        pdf = self.flux_functions[surface]
        dim = len(domain)

        # --- Find an initial guess with flux>0, if x0 is None ---
        if x0 is None:
            for _ in range(max_init_tries):
                trial = np.array([
                    np.random.uniform(low=dom[0], high=dom[1]) for dom in domain
                ])
                if pdf(*trial) > 0:
                    x0 = trial
                    break
            if x0 is None:
                raise ValueError(
                    f"Could not find a positive-flux initial guess after {max_init_tries} tries."
                )
        else:
            # If user-provided x0 is out of domain or flux==0, raise
            for i in range(dim):
                if not (domain[i][0] <= x0[i] <= domain[i][1]):
                    raise ValueError(
                        f"Initial guess x0[{i}]={x0[i]} out of domain {domain[i]}"
                    )
            if pdf(*x0) <= 0:
                raise ValueError(
                    "User-supplied x0 has zero/negative PDF. Please provide a better x0."
                )

        # Evaluate pdf at current point
        f_current = pdf(*x0)
        # If flux is 0 or negative, that is effectively -inf in log
        if f_current <= 0:
            raise ValueError("Initial guess has zero or negative PDF value. Pick a better x0 within domain.")

        # Define proposal standard deviations if not provided
        if proposal_std is None:
            proposal_std = np.array([
                0.1 * (dom[1] - dom[0]) for dom in domain
            ])  # e.g. 10% of domain size

        samples = []
        x_current = x0.copy()

        for _ in range(N):
            # Propose a new candidate point with Gaussian step
            x_proposed = x_current + np.random.normal(scale=proposal_std, size=dim)

            # Check if proposed point is within the domain
            out_of_domain = any(
                (x_proposed[i] < domain[i][0]) or (x_proposed[i] > domain[i][1])
                for i in range(dim)
            )

            if out_of_domain:
                # Immediately reject, keep the old sample
                samples.append(x_current.copy())
                continue

            # Evaluate PDF at proposed
            f_proposed = pdf(*x_proposed)

            if f_proposed <= 0:
                # Also reject
                samples.append(x_current.copy())
                continue

            # Symmetric proposal => acceptance ratio
            alpha = f_proposed / f_current
            if np.random.rand() < alpha:
                # Accept
                x_current = x_proposed
                f_current = f_proposed
            # else reject => keep x_current as is

            samples.append(x_current.copy())

        return np.array(samples)

    def metropolis_hastings_logE(
        self,
        surface: int,
        domain,
        N,
        x0=None,
        proposal_std=None,
        max_init_tries=500
    ):
        """
        MH sampling in (x, w, lnE). domain is [ (x_min, x_max), (w_min, w_max), (lnE_min, lnE_max) ].
        We incorporate the flux*g(E) = flux(x, w, e^{lnE}) * e^{lnE} in acceptance ratio.
        """
        pdf = self.flux_functions[surface]

        def pdf_logE(point):
            x, w, lnE = point
            E = np.exp(lnE)
            return pdf(x, w, E) * E  # incorporate Jacobian

        dim = len(domain)

        # Try multiple times for a positive flux*g(E)
        if x0 is None:
            for _ in range(max_init_tries):
                trial = np.array([
                    np.random.uniform(low=dom[0], high=dom[1]) for dom in domain
                ])
                if pdf_logE(trial) > 0:
                    x0 = trial
                    break
            if x0 is None:
                raise ValueError(
                    f"Could not find a positive-flux*g(E) initial guess after {max_init_tries} tries."
                )
        else:
            # user gave x0
            for i in range(dim):
                if not (domain[i][0] <= x0[i] <= domain[i][1]):
                    raise ValueError(
                        f"Initial guess x0[{i}]={x0[i]} out of domain {domain[i]}"
                    )
            if pdf_logE(x0) <= 0:
                raise ValueError(
                    "User-supplied x0 has zero/negative flux*g(E). Provide a better x0."
                )

        f_current = pdf_logE(x0)
        if f_current <= 0:
            raise ValueError("Initial guess has zero or negative PDF*g(E). Choose a better x0.")

        # Default proposal stdev ~ 10% of domain in each dimension
        if proposal_std is None:
            proposal_std = [
                0.1*(domain[i][1] - domain[i][0]) for i in range(dim)
            ]

        x_current = x0.copy()
        samples = []

        for _ in range(N):
            x_proposed = x_current + np.random.normal(scale=proposal_std, size=dim)

            # Check if proposed is in domain
            out_of_bounds = any(
                (x_proposed[i] < domain[i][0]) or (x_proposed[i] > domain[i][1])
                for i in range(dim)
            )
            if out_of_bounds:
                samples.append(x_current.copy())
                continue

            # Evaluate PDF*g(E) at proposed
            f_proposed = pdf_logE(x_proposed)
            if f_proposed <= 0:
                samples.append(x_current.copy())
                continue

            alpha = f_proposed / f_current
            if np.random.rand() < alpha:
                x_current = x_proposed
                f_current = f_proposed

            samples.append(x_current.copy())

        # Convert last coordinate from lnE -> E
        final = []
        for (xx, ww, lnE) in samples:
            final.append([xx, ww, np.exp(lnE)])
        return np.array(final)


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
        x_bounds, y_bounds, e_bounds = domain
        target_pdf = self.flux_functions[surface]

        # Estimate bounding constant M with uniform sampling
        M = self._estimate_M_uniform(domain, target_pdf, num_trials=50000)

        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        samples_per_worker = [num_samples // num_workers] * num_workers
        for i in range(num_samples % num_workers):
            samples_per_worker[i] += 1

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _rejection_sampling_worker_uniform,
                    x_bounds,
                    y_bounds,
                    e_bounds,
                    n,
                    target_pdf,
                    M,
                )
                for n in samples_per_worker
            ]

            results = [f.result() for f in futures]

        all_samples = np.concatenate(results, axis=0)
        return all_samples
    
    
    def _estimate_M_uniform(self, domain, target_pdf, num_trials=50000):
        """
        Estimate bounding constant M by sampling points uniformly in (x,y,E).
        """
        x_min, x_max = domain[0]
        y_min, y_max = domain[1]
        e_min, e_max = domain[2]

        samples_x = np.random.uniform(x_min, x_max, num_trials)
        samples_y = np.random.uniform(y_min, y_max, num_trials)
        samples_e = np.random.uniform(e_min, e_max, num_trials)

        # Evaluate target flux and proposal pdf
        # proposal_pdf_value = 1 / ((x_max - x_min)*(y_max - y_min)*(e_max - e_min))
        # We'll compute ratio = flux / proposal_pdf_value
        proposal_pdf_val = 1.0 / ((x_max - x_min)*(y_max - y_min)*(e_max - e_min))

        flux_vals = target_pdf(samples_x, samples_y, samples_e)
        ratios = flux_vals / proposal_pdf_val

        # Return a slightly padded max
        return np.max(ratios) * 1.1


    def estimate_max_point(self, surface, grid_points: int = 20) -> float:
        """
        Estimate the location of the maximum value of the flux on a given surface
        """
        space_bounds = config.SPATIAL_BOUNDS[surface]
        angle_bounds = config.ANGULAR_BOUNDS[surface]
        energy_bounds = self.energy_bounds[surface]

        space_vals = np.linspace(space_bounds[0], space_bounds[1], grid_points)
        angle_vals = np.linspace(angle_bounds[0], angle_bounds[1], grid_points)
        energy_vals = np.linspace(energy_bounds[0], energy_bounds[1], grid_points)
        
        pdf_vals = self.evaluate_on_grid(surface, [space_vals, angle_vals, energy_vals])
        max_point = np.argmax(pdf_vals)
        max_point = np.unravel_index(max_point, (grid_points, grid_points, grid_points))
        
        return (space_vals[max_point[0]], angle_vals[max_point[1]], energy_vals[max_point[2]])


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


def _rejection_sampling_worker_uniform(
    x_bounds: List[float],
    y_bounds: List[float],
    e_bounds: List[float],
    num_samples: int,
    target_pdf: Callable[[float, float, float], float],
    M: float
):
    """
    Worker function for generating samples using uniform-based rejection sampling.
    """
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    e_min, e_max = e_bounds

    proposal_pdf_val = 1.0 / ((x_max - x_min)*(y_max - y_min)*(e_max - e_min))

    local_samples = []
    while len(local_samples) < num_samples:
        # Sample uniformly
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        e = np.random.uniform(e_min, e_max)

        fx = target_pdf(x, y, e)
        if fx <= 0:
            # Rejected
            continue

        # Acceptance test
        if np.random.rand() < fx / (M * proposal_pdf_val):
            local_samples.append([x, y, e])

    return np.array(local_samples)