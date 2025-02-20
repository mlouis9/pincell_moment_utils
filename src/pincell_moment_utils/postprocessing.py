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
from pincell_moment_utils.sampling import sample_coefficients
from typing import List, Callable, Union
import numpy as np
from scipy.special import legendre, comb
from scipy.integrate import simpson, quad
from scipy.optimize import nnls
import itertools
import multiprocessing
from abc import ABC, abstractmethod
import warnings

# For convenience
pitch = config.PITCH
TRANSFORM_FUNCTIONS = config.TRANSFORM_FUNCTIONS
WEIGHT_FUNCTIONS = config.WEIGHT_FUNCTIONS
ANGULAR_BOUNDS = config.ANGULAR_BOUNDS
SPATIAL_BOUNDS = config.SPATIAL_BOUNDS

# ----------------------------------------------------------------------
# Simple Bernstein helper
# ----------------------------------------------------------------------
def bernstein_poly(k: int, n: int, x: np.ndarray) -> np.ndarray:
    """
    Compute the Bernstein polynomial B_{k,n}(x) = C(n,k)*x^k*(1 - x)^(n-k),  x in [0,1].
    """
    return comb(n, k)*x**k*(1. - x)**(n - k)

# ----------------------------------------------------------------------
# Surface tally class
# ----------------------------------------------------------------------
class SurfaceMeshTally:
    """A class for postprocessing surface flux mesh tallies"""
    meshes: List[List[np.ndarray]]
    """
    For each of the 4 surfaces, store: [spatial_vals, angle_vals, energy_vals]
    """
    fluxes: List[np.ndarray]
    """
    For each of the 4 surfaces, a numpy array flux[sigma_index, angle_index, energy_index].
    """

    def __init__(self, statepoint_filename: str):
        self.statepoint = openmc.StatePoint(statepoint_filename)
        self.extract_surface_flux()

    def extract_surface_flux(self) -> None:
        self.meshes = []
        self.fluxes = []
        self.energy_filters = []

        for surface_id in range(1, 5):
            tally = self.statepoint.get_tally(id=surface_id)
            svals, avals, evals, efilter = self._extract_meshes(tally)

            # Number of space, angle, and energy points in mesh
            N_space, N_angle, N_energy = len(svals), len(avals), len(evals)

            self.meshes.append([svals, avals, evals])
            self.energy_filters.append(efilter)

            # Filter out high-uncertainty mesh tallies
            flux = tally.mean
            flux_uncert = tally.std_dev
            flux[np.where(flux <= flux_uncert)] = 0.0

            # Reshape
            if surface_id == 2:
                # There's a known extra bin at surface 2
                flux = flux[:, 0, 0].reshape((N_space, N_angle + 1, N_energy), order='C')
                angle_filter = tally.find_filter(openmc.AzimuthalFilter)
                angle_bins = angle_filter.bins
                extra_idx = np.where(angle_bins[:, 0] == -np.pi/2)[0]

                # remove the extra bin
                angle_bins = np.delete(angle_bins, extra_idx, axis=0)
                flux       = np.delete(flux, extra_idx, axis=1)

                # reorder angles
                angle_bins[np.where(angle_bins < 0)] += 2*np.pi
                sorted_indices = np.argsort(angle_bins[:,0])
                flux = flux[:, sorted_indices, :]
            else:
                flux = flux[:, 0, 0].reshape((N_space, N_angle, N_energy), order='C')

            # Normalize to get cell-average flux
            delta_s = pitch / N_space
            delta_a = avals[1] - avals[0]  # uniform spacing
            dE = np.diff(efilter.bins)[:, 0]  # energy bin widths
            flux /= (delta_s * delta_a * dE[np.newaxis, np.newaxis, :])

            self.fluxes.append(flux)

    def _extract_meshes(self, tally) -> List[np.ndarray]:
        """
        Extract (spatial_vals, angle_vals, energy_vals, efilter)
        from the tally object.  We know from problem setup that
        it has a MeshFilter (spatial), an AzimuthalFilter (angle),
        and an EnergyFilter.
        """
        sf = tally.find_filter(openmc.MeshFilter)
        ef = tally.find_filter(openmc.EnergyFilter)
        af = tally.find_filter(openmc.AzimuthalFilter)

        # angle
        if tally.id == 2:
            N_a = len(af.bins) - 1
            angle_bins = np.linspace(np.pi/2, 3*np.pi/2, N_a + 1)
            angle_vals = 0.5*(angle_bins[:-1] + angle_bins[1:])
        else:
            angle_vals = 0.5*(af.values[:-1] + af.values[1:])

        # spatial (manually)
        N_s = len(sf.bins)
        s_bins = np.linspace(-pitch/2, pitch/2, N_s + 1)
        s_vals = 0.5*(s_bins[:-1] + s_bins[1:])

        # energy
        e_vals = 0.5*(ef.values[:-1] + ef.values[1:])

        return [s_vals, angle_vals, e_vals, ef]

# ----------------------------------------------------------------------
# Compute-moments function
# ----------------------------------------------------------------------
def compute_coefficients(mesh_tally, I, J, expansion_type='fourier_legendre'):
    """
    Compute coefficients for either:
    
      1) Fourier–Legendre expansion: 
         shape = (4, I, J, N_energy, 2)
         (Because the basis is orthogonal, direct integration is correct.)
    
      2) Bernstein–Bernstein expansion:
         shape = (4, I, J, N_energy, 1)
         Because Bernstein polynomials are *not* orthogonal, we solve
         the Gram-system (mass-matrix approach).
         BUT, we additionally enforce c >= 0 via nonnegative least squares.
    
    Parameters
    ----------
    mesh_tally : SurfaceMeshTally
        Must have:
         - mesh_tally.meshes[surface] = [sigma_vals, omega_vals, energy_vals]
         - mesh_tally.fluxes[surface] = flux[sigma_idx, omega_idx, E_idx]
    I, J : int
        Number of expansions in the sigma and omega directions.
        For Fourier–Legendre, that means i=0..I-1, j=0..J-1.
        For Bernstein–Bernstein, it also means i=0..I-1, j=0..J-1.
    expansion_type : str
        'fourier_legendre' or 'bernstein_bernstein'.
        
    Returns
    -------
    coefs : np.ndarray
        - fourier_legendre -> shape = (4, I, J, N_energy, 2).
        - bernstein_bernstein -> shape = (4, I, J, N_energy, 1),
          with all coefficients >= 0.
    """
    # pull out some dimension info
    _, _, e_vals = mesh_tally.meshes[0]
    N_energy = len(e_vals)

    # ------------------------------------------------------------------
    # 1) Fourier–Legendre expansion: direct integration
    # ------------------------------------------------------------------
    if expansion_type.lower() == 'fourier_legendre':
        coefs = np.zeros((4, I, J, N_energy, 2))

        # Precompute norm of each basis function
        norm_ijv = np.zeros((4, I, J, 2))
        for sfc in range(4):
            smin, smax = config.SPATIAL_BOUNDS[sfc]
            omin, omax = config.ANGULAR_BOUNDS[sfc]
            length_s = smax - smin
            length_o = omax - omin

            for i in range(I):
                for j in range(J):
                    for v in [0,1]:  # cos vs sin
                        # x-norm
                        if v == 0:  # cos
                            if i == 0:
                                xnorm = length_s
                            else:
                                xnorm = length_s / 2.0
                        else:        # sin
                            if i == 0:
                                xnorm = 0.0
                            else:
                                xnorm = length_s / 2.0

                        # angle-norm for Legendre P_j
                        #  integral_{-1..1} P_j^2 = 2/(2j+1)
                        #  but we have a transform from [omin,omax] => [-1,1],
                        #  so factor = (omax-omin)/(2j+1).
                        onorm = length_o / (2*j + 1) if j >= 0 else 0.0

                        norm_ijv[sfc, i, j, v] = xnorm * onorm

        # Integration to get coefficients
        for sfc in range(4):
            svals, avals, _ = mesh_tally.meshes[sfc]
            flux = mesh_tally.fluxes[sfc]

            # precompute basis on the svals × avals mesh
            basis_cache = _precompute_fourier_legendre_basis(svals, avals, I, J, sfc)

            for i, j, vec_idx in itertools.product(range(I), range(J), [0,1]):
                denom = norm_ijv[sfc, i, j, vec_idx]
                if denom < 1e-14:
                    continue

                B_ij = basis_cache[(i, j, vec_idx)]
                for kE in range(N_energy):
                    product = flux[..., kE] * B_ij
                    raw_integral = simpson(
                        simpson(product, svals, axis=0),
                        avals, axis=0
                    )
                    coefs[sfc, i, j, kE, vec_idx] = raw_integral / denom

        return coefs

    # ------------------------------------------------------------------
    # 2) Bernstein–Bernstein expansion with c >= 0
    # ------------------------------------------------------------------
    elif expansion_type.lower() == 'bernstein_bernstein':
        coefs = np.zeros((4, I, J, N_energy, 1))

        # Precompute the 1D "mass" matrix for Bernstein polynomials on [0,1]
        #   M1D[i_, p_] = \int_0^1 B_{i_,I-1}(x)*B_{p_,I-1}(x) dx
        #   = [comb(I-1,i_)*comb(I-1,p_)] / [(2(I-1)+1)*comb(2(I-1), i_+p_)]
        def bernstein_poly(k, n, x):
            return comb(n, k) * x**k * (1. - x)**(n - k)

        n_deg_s = I-1
        M1D_s = np.zeros((I, I))
        for i_ in range(I):
            for p_ in range(I):
                num = comb(n_deg_s, i_)*comb(n_deg_s, p_)
                den = (2*n_deg_s + 1)*comb(2*n_deg_s, i_+p_)
                M1D_s[i_, p_] = num/den

        n_deg_o = J-1
        M1D_o = np.zeros((J, J))
        for j_ in range(J):
            for q_ in range(J):
                num = comb(n_deg_o, j_)*comb(n_deg_o, q_)
                den = (2*n_deg_o + 1)*comb(2*n_deg_o, j_+q_)
                M1D_o[j_, q_] = num/den

        def scaled_M1D(M1D_base, length):
            # Because if x in [0,1] => original in [x_min, x_max],
            # integral picks up a factor of (x_max - x_min).
            return length * M1D_base

        # For each surface, we build M2D, then solve M2D c = Psi2D
        # under the constraint c >= 0 using NNLS.
        for sfc in range(4):
            svals, avals, _ = mesh_tally.meshes[sfc]
            flux = mesh_tally.fluxes[sfc]

            sigma_min, sigma_max = config.SPATIAL_BOUNDS[sfc]
            omega_min, omega_max = config.ANGULAR_BOUNDS[sfc]

            L_s = sigma_max - sigma_min
            L_o = omega_max - omega_min

            # 2D mass matrix: M2D = kron(M1D_s, M1D_o), each scaled
            Msig = scaled_M1D(M1D_s, L_s)   # shape (I, I)
            Mome = scaled_M1D(M1D_o, L_o)   # shape (J, J)
            size_2D = I*J
            M2D = np.zeros((size_2D, size_2D))
            for i_ in range(I):
                for j_ in range(J):
                    row_idx = i_*J + j_
                    for p_ in range(I):
                        for q_ in range(J):
                            col_idx = p_*J + q_
                            M2D[row_idx, col_idx] = Msig[i_, p_] * Mome[j_, q_]

            # Factor M2D = U^T U for the NNLS transform
            # (Ensure M2D is positive definite; if not, a small tweak or reg. might be needed.)
            U = np.linalg.cholesky(M2D)

            # Precompute the 1D Bernstein polynomials for svals, avals
            tsigma = (svals - sigma_min)/L_s  # scaled to [0,1]
            tomega = (avals - omega_min)/L_o
            Bsig = np.zeros((I, len(svals)))
            Bome = np.zeros((J, len(avals)))
            for i_ in range(I):
                Bsig[i_, :] = bernstein_poly(i_, I-1, tsigma)
            for j_ in range(J):
                Bome[j_, :] = bernstein_poly(j_, J-1, tomega)

            # For each energy bin => build Psi2D then do nonnegative least squares
            for kE in range(N_energy):
                # Build Psi2D by integrating flux * B_{i}(sigma)* B_{j}(omega)
                Psi2D = np.zeros(size_2D)
                for i_ in range(I):
                    for j_ in range(J):
                        row_idx = i_*J + j_
                        basis_2D = np.outer(Bsig[i_, :], Bome[j_, :])
                        product_2D = flux[..., kE]*basis_2D

                        val = simpson(
                            simpson(product_2D, svals, axis=0),
                            avals, axis=0
                        )
                        Psi2D[row_idx] = val

                # We want to solve: min_{c >= 0}  1/2 ||U c - b||^2
                # where b = U^{-T} Psi2D, so that the gradient is M2D c - Psi2D = 0
                weights = np.linalg.norm(M2D, axis=1)  # Column-wise norms
                M2D_weighted = M2D / weights[:, np.newaxis]
                Psi2D_weighted = Psi2D / weights

                c2D, _ = nnls(M2D_weighted, Psi2D_weighted)


                # Reshape back
                for i_ in range(I):
                    for j_ in range(J):
                        idx_ = i_*J + j_
                        coefs[sfc, i_, j_, kE, 0] = c2D[idx_]

        return coefs

    else:
        raise ValueError(f"Unknown expansion_type: {expansion_type}")
def _precompute_fourier_legendre_basis(
    svals: np.ndarray, avals: np.ndarray,
    I: int, J: int, surface: int
) -> dict:
    """
    Precompute cos/sin in x and Legendre in angle for i=0..I-1, j=0..J-1,
    plus vector_index = 0 or 1 (cos vs sin).
    """
    tf = TRANSFORM_FUNCTIONS[surface]
    basis = {}
    for i, j in itertools.product(range(I), range(J)):
        # cos
        cos_s = np.cos(i*np.pi*svals[:,None]/(pitch/2.))
        L_j   = legendre(j)(tf(avals))[None,:]
        basis[(i, j, 0)] = cos_s*L_j

        # sin
        sin_s = np.sin(i*np.pi*svals[:,None]/(pitch/2.))
        basis[(i, j, 1)] = sin_s*L_j

    return basis


# ----------------------------------------------------------------------
# Base expansion classes
# ----------------------------------------------------------------------
class SurfaceExpansionBase(ABC):
    """
    Common base for either FourierLegendreExpansion or BernsteinBernsteinExpansion.
    We store:
        shape = (4, I, J, N_energy, N_vector)
    so that for Fourier-Legendre, N_vector=2; for Bernstein-Bernstein, N_vector=1.
    """
    def __init__(self, coefficients: np.ndarray, energy_filters: list):
        # e.g. shape = (4, I, J, N_energy, 2)
        shape = coefficients.shape
        self._n_surfaces     = shape[0]
        self._n_spatial_terms= shape[1]
        self._n_angular_terms= shape[2]
        self._n_energy       = shape[3]
        self._n_vector       = shape[4]

        self.coefficients   = coefficients
        self.energy_filters = energy_filters

        # Build expansions for each surface
        self.flux_functions = [
            self._construct_surface_expansion(sfc)
            for sfc in range(4)
        ]
        self.energy_bounds = [
            [f.bins[0,0], f.bins[-1,1]] for f in energy_filters
        ]

    @abstractmethod
    def _construct_surface_expansion(self, surface: int) -> Callable:
        pass

    @abstractmethod
    def _precompute_basis_functions(
        self, svals: np.ndarray,
        avals: np.ndarray,
        surface: int
    ) -> dict:
        pass

    @abstractmethod
    def _integral_basis_function(
        self, i: int, j: int, v: int, surface: int
    ) -> Callable:
        pass

    def integrate_flux(self, surface: int) -> float:
        """
        Integrate flux over the entire domain of (sigma,omega) and over all energy bins.
        """
        smin, smax = SPATIAL_BOUNDS[surface]
        omin, omax = ANGULAR_BOUNDS[surface]

        ef = self.energy_filters[surface]
        dE = np.diff(ef.bins, axis=1).flatten()
        N_eBins = len(dE)

        total = 0.0
        for e_idx in range(N_eBins):
            # sum up expansions in space+angle
            local_val = 0.0
            for i, j, v in itertools.product(
                range(self._n_spatial_terms),
                range(self._n_angular_terms),
                range(self._n_vector)
            ):
                c_ijv = self.coefficients[surface, i, j, e_idx, v]
                # integral of basis:
                f_int = self._integral_basis_function(i, j, v, surface)
                local_val += c_ijv*f_int(smin, smax, omin, omax)

            total += local_val*dE[e_idx]

        return total

    def normalize_by(self, norm: Union[float, list[float]]) -> None:
        """
        Multiply coefficients by 1/norm, either globally or per-surface.
        """
        if isinstance(norm, float):
            self.coefficients /= norm
        else:
            # norm is list/array of length 4
            for s in range(4):
                self.coefficients[s] /= norm[s]

        # rebuild expansions
        self.flux_functions = [
            self._construct_surface_expansion(sfc)
            for sfc in range(4)
        ]

    def evaluate_on_grid(
        self, surface: int,
        grid_points: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        Evaluate flux at each (sigma,omega,E) in a structured 3D grid.
        Returns flux[ Ns, Nw, NE ].
        """
        svals, avals, Evals = grid_points
        Ns, Nw, NE = len(svals), len(avals), len(Evals)
        out = np.zeros((Ns, Nw, NE))

        # precompute basis on 2D mesh of (svals, avals)
        basis_cache = self._precompute_basis_functions(svals, avals, surface)

        for i, j, v in itertools.product(
            range(self._n_spatial_terms),
            range(self._n_angular_terms),
            range(self._n_vector)
        ):
            # For each energy in Evals, find bin and add contribution
            for kE, E in enumerate(Evals):
                e_idx = self._find_energy_bin(surface, E)
                coef = self.coefficients[surface, i, j, e_idx, v]
                out[..., kE] += coef*basis_cache[(i,j,v)]

        return np.maximum(out, 0.)

    def _find_energy_bin(self, surface: int, E: float) -> int:
        """
        Return index of the energy bin that contains E.
        """
        e_bins = self.energy_filters[surface].bins
        for idx, (E0,E1) in enumerate(e_bins):
            if E0 <= E <= E1:
                return idx
        return 0  # fallback

    def generate_samples(
        self, N: int, sample_surface=None,
        num_cores: int = multiprocessing.cpu_count(),
        method: str = 'ensemble', use_log_energy: bool = True,
        burn_in: int = 1000, progress=False
    ):
        """
        High-level API for sampling from these expansions.  Relies on your
        `sample_surface_flux()` function externally.
        """
        from pincell_moment_utils.sampling import sample_surface_flux

        if sample_surface is not None:
            surfaces = [sample_surface]
        else:
            surfaces = [0,1,2,3]

        # 1) compute integrals => normalizations
        norms = np.ones(4)
        for sfc in surfaces:
            norms[sfc] = self.integrate_flux(sfc)

        # 2) normalize
        self.normalize_by(norms)

        # 3) partition N among surfaces
        N_surf = np.floor(N*norms/np.sum(norms)).astype(int)
        N_surf[-1] += N - np.sum(N_surf)

        # 4) generate samples
        all_samples = [[] for _ in range(4)]
        for sfc in surfaces:
            sdom = config.SPATIAL_BOUNDS[sfc]
            wdom = config.ANGULAR_BOUNDS[sfc]
            ebnd = self.energy_bounds[sfc]

            if use_log_energy:
                domain = [sdom, wdom, (np.log(ebnd[0]), np.log(ebnd[1]))]
            else:
                domain = [sdom, wdom, (ebnd[0], ebnd[1])]

            pdf = self.flux_functions[sfc]
            needed = int(N_surf[sfc]) if sample_surface is None else N

            samples = sample_surface_flux(
                pdf=pdf, domain=domain, N=needed,
                method=method, use_log_energy=use_log_energy,
                burn_in=burn_in, num_cores=num_cores, progress=progress
            )
            all_samples[sfc] = samples

        # 5) un‐normalize to restore original scale
        self.normalize_by(1.0/norms)

        return all_samples

    def estimate_max_point(self, surface: int, grid_points=20):
        """
        Coarse search for max flux in a uniform (space, angle, E) mesh.
        """
        smin, smax = SPATIAL_BOUNDS[surface]
        omin, omax = ANGULAR_BOUNDS[surface]
        ebnd = self.energy_bounds[surface]

        svals = np.linspace(smin, smax, grid_points)
        avals = np.linspace(omin, omax, grid_points)
        Evals = np.linspace(ebnd[0], ebnd[1], grid_points)

        pdf_vals = self.evaluate_on_grid(surface, (svals, avals, Evals))
        max_idx = np.argmax(pdf_vals)
        iS, iA, iE = np.unravel_index(max_idx, pdf_vals.shape)
        return (svals[iS], avals[iA], Evals[iE])


# ----------------------------------------------------------------------
# Fourier-Legendre Expansion
# ----------------------------------------------------------------------
class FourierLegendreExpansion(SurfaceExpansionBase):
    def _construct_surface_expansion(self, surface: int) -> Callable:
        return _FourierLegendreReconstructedFlux(
            self.coefficients, self.energy_filters,
            surface, self._n_spatial_terms, self._n_angular_terms
        )

    def _precompute_basis_functions(self, svals, avals, surface: int) -> dict:
        out = {}
        tf = TRANSFORM_FUNCTIONS[surface]
        pitch_half = pitch/2.
        for i, j, v in itertools.product(
            range(self._n_spatial_terms),
            range(self._n_angular_terms),
            range(self._n_vector)
        ):
            # v=0 => cos, v=1 => sin
            angle_part = legendre(j)(tf(avals))[None,:]
            if v == 0:
                space_part = np.cos(i*np.pi*svals[:,None]/pitch_half)
            else:
                space_part = np.sin(i*np.pi*svals[:,None]/pitch_half)
            out[(i,j,v)] = space_part*angle_part
        return out

    def _integral_basis_function(self, i: int, j: int, v: int, surface: int):
        tf = TRANSFORM_FUNCTIONS[surface]
        pitch_half = pitch/2.

        def integrand_angle(w):
            return legendre(j)(tf(w))

        if v == 0:
            # cos
            def f_int(xl, xu, wl, wu):
                if i == 0:
                    x_part = (xu - xl)
                else:
                    k = i*np.pi/pitch_half
                    x_part = (1./k)*(np.sin(k*xu) - np.sin(k*xl))
                w_part, _ = quad(integrand_angle, wl, wu)
                return x_part*w_part
        else:
            # sin
            def f_int(xl, xu, wl, wu):
                if i == 0:
                    return 0.0
                k = i*np.pi/pitch_half
                x_part = (1./k)*(-np.cos(k*xu) + np.cos(k*xl))
                w_part, _ = quad(integrand_angle, wl, wu)
                return x_part*w_part

        return f_int


class _FourierLegendreReconstructedFlux:
    def __init__(self, coefs, efilters, surface, I, J):
        self.coefs = coefs
        self.efilters = efilters
        self.surface = surface
        self.I = I
        self.J = J

    def __call__(self, x, w, E):
        # find energy bin
        e_idx = 0
        for idx, (E0, E1) in enumerate(self.efilters[self.surface].bins):
            if E0 <= E <= E1:
                e_idx = idx
                break

        tf = TRANSFORM_FUNCTIONS[self.surface]
        val = 0.0
        pitch_half = pitch/2.

        for i, j, v in itertools.product(range(self.I), range(self.J), range(2)):
            c_ijv = self.coefs[self.surface, i, j, e_idx, v]
            if v == 0:
                # cos
                val += c_ijv * np.cos(i*np.pi*x/pitch_half)*legendre(j)(tf(w))
            else:
                # sin
                val += c_ijv * np.sin(i*np.pi*x/pitch_half)*legendre(j)(tf(w))

        return max(val, 0.0)


def _angle_mod_2pi(angles: np.ndarray) -> np.ndarray:
    """
    Convert angles (which may be negative or >2π) into [0,2π).
    """
    return np.mod(angles, 2.0*np.pi)


def clamp_array_and_warn(
    x: np.ndarray,
    x_min: float,
    x_max: float,
    surface: int,
    kind: str = "angle"
) -> np.ndarray:
    """
    Vectorized clamp of array 'x' to [x_min, x_max].
    Warn whenever we actually clamp any values.
    """
    x_clamped = x.copy()
    mask_low  = (x_clamped < x_min)
    mask_high = (x_clamped > x_max)

    n_low  = np.count_nonzero(mask_low)
    n_high = np.count_nonzero(mask_high)

    if n_low > 0:
        example_val = x_clamped[mask_low][0]
        warnings.warn(
            f"{n_low} {kind}(s) out of domain on surface {surface}; "
            f"example={example_val:.4g}, clamping to {x_min}.",
            stacklevel=2
        )
        x_clamped[mask_low] = x_min

    if n_high > 0:
        example_val = x_clamped[mask_high][0]
        warnings.warn(
            f"{n_high} {kind}(s) out of domain on surface {surface}; "
            f"example={example_val:.4g}, clamping to {x_max}.",
            stacklevel=2
        )
        x_clamped[mask_high] = x_max

    return x_clamped

def clamp_scalar_and_warn(
    x: float,
    x_min: float,
    x_max: float,
    surface: int,
    kind: str = "angle"
) -> float:
    """
    Scalar clamp with a single warning if we go out of [x_min, x_max].
    """
    if x < x_min:
        warnings.warn(
            f"{kind}={x:.4g} out of domain on surface {surface}; "
            f"clamping to {x_min}.",
            stacklevel=2
        )
        return x_min
    elif x > x_max:
        warnings.warn(
            f"{kind}={x:.4g} out of domain on surface {surface}; "
            f"clamping to {x_max}.",
            stacklevel=2
        )
        return x_max
    else:
        return x



# ----------------------------------------------------------------------
# Bernstein–Bernstein Expansion
# ----------------------------------------------------------------------
class BernsteinBernsteinExpansion(SurfaceExpansionBase):
    """
    Expects shape = (4, I, J, N_energy, 1).
    We'll interpret i=0..I-1 => B_{i,I-1}, j=0..J-1 => B_{j,J-1}.
    """
    def _construct_surface_expansion(self, surface: int) -> Callable:
        return _BernsteinBernsteinReconstructedFlux(
            self.coefficients, self.energy_filters,
            surface, self._n_spatial_terms, self._n_angular_terms
        )

    def _precompute_basis_functions(self, svals: np.ndarray, avals: np.ndarray, surface: int) -> dict:
        out = {}
        smin, smax = SPATIAL_BOUNDS[surface]
        omin, omax = ANGULAR_BOUNDS[surface]

        # Clamp to [omin, omax], with a warning if out-of-range
        avals_clamped = clamp_array_and_warn(avals, omin, omax, surface=surface, kind="angle")

        # Then map into [0,1]
        tsigma = (svals - smin)/(smax - smin)
        tomega = (avals_clamped - omin)/(omax - omin)

        # Finally, ensure no floating slop outside [0,1].
        tomega = np.clip(tomega, 0.0, 1.0)

        I = self._n_spatial_terms
        J = self._n_angular_terms

        Bsig = np.zeros((I, len(svals)))
        Bome = np.zeros((J, len(avals)))

        for i_ in range(I):
            Bsig[i_, :] = bernstein_poly(i_, I-1, tsigma)
        for j_ in range(J):
            Bome[j_, :] = bernstein_poly(j_, J-1, tomega)

        for i_, j_ in itertools.product(range(I), range(J)):
            out[(i_, j_, 0)] = np.outer(Bsig[i_, :], Bome[j_, :])
        return out



    def _integral_basis_function(self, i: int, j: int, v: int, surface: int):
        """
        For a full-domain integral of B_{i,I-1} * B_{j,J-1} in sigma,omega.
        We'll use the closed-form integral = (smax-smin)/I * (omax-omin)/J
        if the integration bounds match the entire domain. Otherwise fallback
        to numeric.
        """
        smin, smax = SPATIAL_BOUNDS[surface]
        omin, omax = ANGULAR_BOUNDS[surface]

        I = self._n_spatial_terms
        J = self._n_angular_terms

        def f_int(xl, xu, wl, wu):
            xl_ = max(xl, smin)
            xu_ = min(xu, smax)
            wl_ = max(wl, omin)
            wu_ = min(wu, omax)

            full_s = (abs(xl_ - smin)<1e-13 and abs(xu_ - smax)<1e-13)
            full_w = (abs(wl_ - omin)<1e-13 and abs(wu_ - omax)<1e-13)

            if full_s and full_w:
                # closed form
                return ((smax - smin)/I)*((omax - omin)/J)
            else:
                # fallback numeric
                Ns, Nw = 50, 50
                svals = np.linspace(xl_, xu_, Ns)
                wvals = np.linspace(wl_, wu_, Nw)
                tsig = (svals - smin)/(smax - smin)
                tome = (wvals - omin)/(omax - omin)

                # precompute polynomials
                # B_{i,I-1}(tsig), B_{j,J-1}(tome)
                Bi = bernstein_poly(i, I-1, tsig)
                Bj = bernstein_poly(j, J-1, tome)

                mesh = np.zeros((Ns, Nw))
                for is_ in range(Ns):
                    for iw in range(Nw):
                        mesh[is_, iw] = Bi[is_]*Bj[iw]
                ds = (xu_ - xl_)/(Ns-1)
                dw = (wu_ - wl_)/(Nw-1)
                return mesh.sum()*ds*dw

        return f_int
    
    def integrate_flux(self, surface: int) -> float:
        """
        Integrate the flux over the *entire* domain in sigma x omega x E
        using the known fact that
          ∫ B_{i,I-1}(\tildeσ) dσ over [σ_min,σ_max] = (σ_max - σ_min)/I
        and similarly for omega in [ω_min,ω_max],
        plus the bin widths for energy.
        
        This direct summation is valid *only* for the *full* domain.
        """
        # 1) Pull domain bounds
        smin, smax = SPATIAL_BOUNDS[surface]
        omin, omax = ANGULAR_BOUNDS[surface]

        # 2) Pull the energy bin widths
        efilter = self.energy_filters[surface]
        dE = np.diff(efilter.bins, axis=1).flatten()  # shape (N_energy,)

        # 3) The total scale factor for space+angle:
        #    Because integral of B_{i,I-1} is 1/I on [0,1],
        #    after scaling to [smin, smax], we get (smax-smin)/I.
        #    Similarly (omax-omin)/J for the angle dimension.
        space_factor = ((smax - smin)/self._n_spatial_terms) * \
                       ((omax - omin)/self._n_angular_terms)

        # 4) Our coefficients for this surface: shape (I, J, N_energy, 1)
        c_ijE = self.coefficients[surface, :, :, :, 0]  # => shape (I, J, N_energy)

        # 5) Sum over i,j => sum_c[k] = Σ_{i=0..I-1} Σ_{j=0..J-1} c_{i,j,k}
        sum_c_k = c_ijE.sum(axis=(0,1))  # shape (N_energy,)

        # 6) Multiply each sum_c_k by (space_factor * dE[k]) and sum over k
        flux_integral = np.sum(sum_c_k * space_factor * dE)
        return flux_integral


class _BernsteinBernsteinReconstructedFlux:
    def __init__(self, coefs, efilters, surface, I, J):
        self.coefs = coefs
        self.efilters = efilters
        self.surface = surface
        self.I = I
        self.J = J
        self.smin, self.smax = SPATIAL_BOUNDS[surface]
        self.omin, self.omax = ANGULAR_BOUNDS[surface]

    def __call__(self, sigma, omega, E):
        # 1) find the energy bin
        e_idx = 0
        for idx, (E0, E1) in enumerate(self.efilters[self.surface].bins):
            if E0 <= E <= E1:
                e_idx = idx
                break

        # 2) clamp directly to [omin, omax], warning if out-of-range
        omega_clamped = clamp_scalar_and_warn(
            omega, self.omin, self.omax,
            surface=self.surface, kind="angle"
        )

        # 3) scale into [0,1]
        tsig = (sigma - self.smin)/(self.smax - self.smin)
        tome = (omega_clamped - self.omin)/(self.omax - self.omin)

        # 4) evaluate
        val = 0.0
        for i_, j_ in itertools.product(range(self.I), range(self.J)):
            c_ij = self.coefs[self.surface, i_, j_, e_idx, 0]
            val += c_ij * bernstein_poly(i_, self.I-1, tsig)*bernstein_poly(j_, self.J-1, tome)

        return max(val, 0.0)

# ----------------------------------------------------------------------
# Factory function
# ----------------------------------------------------------------------
def surface_expansion(
    coefficients: np.ndarray,
    energy_filters: list,
    expansion_type: str = 'bernstein_bernstein'
) -> SurfaceExpansionBase:
    """
    Build either FourierLegendreExpansion or BernsteinBernsteinExpansion,
    given the 5D coefficient array:
       shape = (4, I, J, N_energy, 2) for FourierLegendre,
                (4, I, J, N_energy, 1) for BernsteinBernstein.
    """
    expansion_type = expansion_type.lower()
    if expansion_type == 'fourier_legendre':
        return FourierLegendreExpansion(coefficients, energy_filters)
    elif expansion_type == 'bernstein_bernstein':
        return BernsteinBernsteinExpansion(coefficients, energy_filters)
    else:
        raise ValueError(f"Unknown expansion_type: {expansion_type}")


def random_surface_expansion(num_space: int, num_angle: int, energy_filters: list, 
                             expansion_type: str = 'bernstein_bernstein') -> SurfaceExpansionBase:
    """Create a surface expansion with randomly sampled coefficients. This surface expansion will be properly normalized, and
    have coefficients that ensure a positive definite surface expansion. This is currently only implemented for the Bernstein Bernstein
    expansion.
    
    Parameters
    ----------
    num_space
        The number of spatial coefficients in the expansion
    num_angle
        The number of angular coefficients in the expansion
    energy_filters
        The energy filters to use for the expansion
    expansion_type
        The type of expansion to use. Currently only 'bernstein_bernstein' is supported.

    Returns
    -------
    SurfaceExpansionBase
        The surface expansion with randomly sampled coefficients.
    """


    if expansion_type == 'bernstein_bernstein':
        # Assuming same number of energy bins for each surface
        shape = (4, num_space, num_angle, energy_filters[0].bins.shape[0], 1)
        coefficients = np.zeros(shape)
        for surface in range(4):
            coefficients[surface] = sample_coefficients(shape[1:])
        
        # Now rescale coefficients according to the functional expansion
        for surface in range(4):
            dE = np.diff(energy_filters[surface].bins, axis=1).flatten()
            L_σ = SPATIAL_BOUNDS[surface][1] - SPATIAL_BOUNDS[surface][0]
            L_ω = ANGULAR_BOUNDS[surface][1] - ANGULAR_BOUNDS[surface][0]
            coefficients[surface, :, :, :, :] /= ( dE[np.newaxis, np.newaxis, :, np.newaxis]*(L_σ/(num_space)) * (L_ω/(num_angle)) )

        return BernsteinBernsteinExpansion(coefficients, energy_filters)
    else:
        raise ValueError(f"Unknown expansion_type: {expansion_type}")