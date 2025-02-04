from pincell_moment_utils import postprocessing as pp
from pincell_moment_utils import config
import openmc
import numpy as np
import multiprocessing

N_samples = 1E+07

# First extract the surface fluxes from the tallies, then use them to compute the moments of the expansion 
mesh_tally = pp.SurfaceMeshTally('../data/source_statepoint.100.h5')
coefficients = pp.compute_moments(mesh_tally, 7, 5)

expansion = pp.SurfaceExpansion(coefficients, mesh_tally.energy_filters)
space_vals, angle_vals, energy_vals = mesh_tally.meshes[0]
expansion_vals = expansion.evaluate_on_grid(0, (space_vals, angle_vals, energy_vals))

samples = expansion.generate_samples(N_samples, num_cores=multiprocessing.cpu_count(), burn_in=1000, progress=True)

pitch = config.PITCH
surface_perpendicular_coordinate = [pitch/2, -pitch/2, pitch/2, -pitch/2]
surface_coord_to_3d = [ 
    lambda x: (surface_perpendicular_coordinate[0], x, 0),
    lambda x: (surface_perpendicular_coordinate[1], x, 0),
    lambda x: (x, surface_perpendicular_coordinate[2], 0),
    lambda x: (x, surface_perpendicular_coordinate[3], 0)
]

source_particles = []
for surface in range(4):
    for sample in samples[surface]:
        p = openmc.SourceParticle(r=surface_coord_to_3d[surface](sample[0]), u = (np.cos(sample[1]), np.sin(sample[1]), 0), 
                              E=sample[2])
        source_particles.append(p)

openmc.write_source_file(source_particles, f"incident_flux.h5")