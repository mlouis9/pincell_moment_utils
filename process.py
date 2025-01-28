import openmc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # Import LogNorm for logarithmic scaling

# Load statepoint file
pitch = 1.26
N = 15
sp = openmc.StatePoint('statepoint.100.h5')  # Adjust filename based on batches

# Get flux tally
flux_tally = sp.get_tally(name='flux')
flux_data = flux_tally.get_slice(scores=['flux']).mean.reshape((N, N))

# Get fission density tally
fission_tally = sp.get_tally(name='fission')
fission_data = fission_tally.get_slice(scores=['fission']).mean.reshape((N, N))

# Replace zeros or negative values with a small positive value for logarithmic scaling
flux_data[flux_data <= 0] = 1e-10
fission_data[fission_data <= 0] = 1e-10

# Create plots
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot flux with logarithmic scale
flux_plot = ax[0].imshow(flux_data, origin='lower', extent=[-pitch/2, pitch/2, -pitch/2, pitch/2],
                         cmap='viridis', norm=LogNorm(vmin=flux_data.min(), vmax=flux_data.max()))
ax[0].set_title('Flux (Log Scale)')
ax[0].set_xlabel('X [cm]')
ax[0].set_ylabel('Y [cm]')
fig.colorbar(flux_plot, ax=ax[0])

# Plot fission density with logarithmic scale
fission_plot = ax[1].imshow(fission_data, origin='lower', extent=[-pitch/2, pitch/2, -pitch/2, pitch/2],
                            cmap='inferno', norm=LogNorm(vmin=fission_data.min(), vmax=fission_data.max()))
ax[1].set_title('Fission Density (Log Scale)')
ax[1].set_xlabel('X [cm]')
fig.colorbar(fission_plot, ax=ax[1])

# Save the figure
plt.tight_layout()
plt.savefig('flux_and_fission_density_logscale.png', dpi=300)
plt.show()