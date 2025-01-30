import numpy as np
import matplotlib.pyplot as plt
import openmc
from matplotlib.colors import LogNorm

pitch = 1.26

# Load tally results from OpenMC statepoint file
statepoint = openmc.StatePoint("statepoint.1000.h5")  # Adjust batch number as needed

# Extract the tally of interest
tally = statepoint.get_tally(name="flux_at_right_boundary")

# Get tally data
flux = tally.mean  # The flux values (mean)
flux_uncertainty = tally.std_dev  # The standard deviation

# Extract filters for normalization
mesh_filter = tally.find_filter(openmc.MeshFilter)
energy_filter = tally.find_filter(openmc.EnergyFilter)
angle_filter = tally.find_filter(openmc.AzimuthalFilter)

E_min = energy_filter.bins[0][0]
E_max = energy_filter.bins[-1][-1]

Ny = len(mesh_filter.bins)
NE = len(energy_filter.bins)
Nω = len(angle_filter.bins)

flux = flux[:, 0, 0].reshape([Ny, Nω, NE], order='C')

# Get bin widths for normalization
delta_y = pitch/Ny
delta_omega = angle_filter.bins[0][1] - angle_filter.bins[0][0]
delta_E = np.diff(energy_filter.bins)[:, 0]  # Energy group widths

# Normalize the flux
flux_normalized = flux / (delta_y * delta_omega * delta_E[np.newaxis, np.newaxis, :])

# Select a representative slice for each variable
# Example: fixing energy group, angle, or y-axis index
energy_index = len(delta_E) // 4  # Representative energy group (middle of the spectrum)
angle_index = len(angle_filter.bins) // 2  # Representative angle (middle bin)
mesh_index = Ny // 2  # Representative y-axis index (middle of the mesh)

# Plot a 2D slice with energy fixed
flux_energy_slice = flux_normalized[:, :, energy_index]  # Fix energy
print(flux_energy_slice.shape)
plt.figure(figsize=(8, 6))
plt.imshow(flux_energy_slice.T, origin="lower", extent=[-pitch/2, pitch/2, -np.pi/2, np.pi/2], aspect="auto",norm=LogNorm())
plt.colorbar(label="Flux (normalized)")
plt.xlabel("y-position")
plt.ylabel("Angle (rad)")
plt.title(f"Flux Slice at Energy {(energy_filter.bins[energy_index][1] - energy_filter.bins[energy_index][0])/2} eV")
plt.show()

# Plot a 2D slice with angle fixed
flux_angle_slice = flux_normalized[:, angle_index, :]  # Fix angle
plt.figure(figsize=(8, 6))
plt.imshow(flux_angle_slice.T, origin="lower", extent=[-pitch/2, pitch/2, E_min, E_max], aspect="auto",norm=LogNorm())
plt.colorbar(label="Flux (normalized)")
plt.ylabel("Energy (eV)")
plt.xlabel("y-position")
plt.title(f"Flux Slice at Angle {(angle_filter.bins[angle_index][1] - angle_filter.bins[angle_index][0])/2} radians")
plt.show()

# Plot a 2D slice with y fixed
flux_y_slice = flux_normalized[mesh_index, :, :]  # Fix y location
plt.figure(figsize=(8, 6))
plt.imshow(flux_y_slice.T, origin="lower", extent=[-np.pi/2, np.pi/2, E_min, E_max], aspect="auto",norm=LogNorm())
plt.colorbar(label="Flux (normalized)")
plt.ylabel("Energy (eV)")
plt.xlabel("Angle (rad)")
plt.title(f"Flux Slice at y-position Index {mesh_index}")
plt.show()