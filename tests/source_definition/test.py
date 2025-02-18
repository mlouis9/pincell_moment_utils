from pincell_moment_utils import postprocessing as pp
import time

# First extract the surface fluxes from the tallies, then use them to compute the moments of the expansion 
mesh_tally = pp.SurfaceMeshTally('../data/source_statepoint.100.h5')
coefficients = pp.compute_moments(mesh_tally, 7, 5)

expansion = pp.SurfaceExpansion(coefficients, mesh_tally.energy_filters)

start = time.time()
samples = expansion.generate_samples(1000, surface=0, num_cores=8, method='ensemble', burn_in=100, use_log_energy=True, progress=True)
stop = time.time()
print([len(samples[surface]) for surface in range(4)])