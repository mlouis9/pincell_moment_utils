"""
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
from typing import List, Callable
import numpy as np
from scipy.special import legendre
from scipy.integrate import simpson
from itertools import product

pitch = config.PITCH
TRANSFORM_FUNCTIONS = config.TRANSFORM_FUNCTIONS
WEIGHT_FUNCTIONS = config.WEIGHT_FUNCTIONS

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
        for id in range(1, 5):
            tally = self.statepoint.get_tally(id=id)
            spatial_vals, angle_vals, energy_vals, energy_filter = self.extract_meshes(tally)

            # Number of space, angle, and energy points in mesh
            N_space, N_angle, N_energy = len(spatial_vals), len(angle_vals), len(energy_vals)

            self.meshes.append([spatial_vals, angle_vals, energy_vals])

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

    def extract_meshes(self, tally) -> List[np.ndarray]:
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

    coefs = np.zeros((4, I, J, N_energy, 2)) # Note for this particular expansion, coefficients are vectors in R^2, hence the last 2
    
    for surface in range(4):
        space_vals, angle_vals, energy_vals = mesh_tally.meshes[0]
        flux = mesh_tally.fluxes[surface]

        for i, j, vector_index in product(range(I), range(J), range(2)):
            # Evaluate integrand functions on spatial and angular mesh
            integrand_function = _integrand_functions(i, j, vector_index, SurfaceMeshTally)
            integrand_eval = _evaluate_integrand(integrand_function, space_vals, angle_vals)
            
            for k in range(N_energy):
                # Multiply the basis functions with the flux
                integrand_eval *= flux[:, :, k]
                
                # Compute and assign the coefficient
                coefs[surface, i, j, k, vector_index] = _compute_integral(integrand_eval, i, j, vector_index, space_vals, angle_vals)

    return coefs


def _integrand_functions(i: int, j: int, vector_index: int, surface: int) -> Callable[[float, float], float]:
    """Returns the basis function corresponding to a given i, j, and vector index in the function expansion"""

    weight_function = WEIGHT_FUNCTIONS[surface]
    transform_function = TRANSFORM_FUNCTIONS[surface]
    if vector_index == 0:
        def integrand(x: float, ω: float) -> float:
            return np.cos(i*np.pi* x/(pitch/2))*legendre(j)(transform_function(ω))*weight_function(ω)
    elif vector_index == 1:
        def integrand(x: float, ω: float) -> float:
            return np.sin(i*np.pi* x/(pitch/2))*legendre(j)(transform_function(ω))*weight_function(ω)
    else:
        raise ValueError(f"vector_index must be 0 or 1, you supplied {vector_index}")
    return integrand

def _evaluate_integrand(integrand_function: Callable[[float, float], float], space_vals, angle_vals):
    integrand_eval = np.zeros((len(space_vals), len(angle_vals)))
    for y_idx, x in enumerate(space_vals):
        for ω_idx, ω in enumerate(angle_vals):
            integrand_eval[y_idx, ω_idx] = integrand_function(x, ω)
    return integrand_eval

def _compute_integral(integrand_eval, i, j, vector_index, space_vals, angle_vals):
    # Base case: integral over cosine for zero angular frequency
    if i == 0:
        if vector_index == 0:
            return (2 * j + 1) / (2 * pitch) * simpson(simpson(integrand_eval, space_vals, axis=0), angle_vals, axis=0)
        else:
            return 0  # Sine basis integral is zero
    else:
        return (2 * j + 1) / pitch * simpson(simpson(integrand_eval, space_vals, axis=0), angle_vals, axis=0)
