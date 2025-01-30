import openmc
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import Legendre
import scipy.integrate

pitch = 1.26

# Load the StatePoint file
sp = openmc.StatePoint('statepoint.100.h5')  # Ensure the filename matches your output

# Retrieve the mesh tally for flux
mesh_tally = sp.get_tally(name="flux_at_right_boundary")

# Extract the flux values (mean) and uncertainties (std. dev.)
mesh_flux = mesh_tally.mean.flatten()  # Flatten to 1D array
mesh_flux_uncertainty = mesh_tally.std_dev.flatten()  # Standard deviation of flux

# Define the y-bin edges based on the mesh
y_bins = np.linspace(-pitch/2, pitch/2, len(mesh_flux) + 1)  # Adjust for pitch and number of bins
Δy = y_bins[1] - y_bins[0]
y_bin_centers = 0.5 * (y_bins[:-1] + y_bins[1:])  # Compute bin centers for plotting

# Retrieve the Legendre moment tally
moment_tally = sp.get_tally(name="flux_moment_at_right_boundary")

# Extract the Legendre moments (mean) and uncertainties
legendre_moments = moment_tally.mean.flatten()  # Flatten to 1D array
legendre_uncertainties = moment_tally.std_dev.flatten()  # Standard deviation of moments

for i in range(len(legendre_moments)):
    if abs(legendre_moments[i]) <= legendre_uncertainties[i]:
        legendre_moments[i] = 0.0

print(legendre_moments)

# Define the Legendre polynomial coefficients
# The coefficients are given by (2n + 1)/2 * moment for each order n
order = len(legendre_moments) - 1  # Determine the highest order of the moments
coefficients = (2 * np.arange(order + 1) + 1) / pitch * legendre_moments

# Reconstruct the flux using the Legendre polynomial
legendre_poly = Legendre(coefficients, domain=[-pitch / 2, pitch / 2])  # Define the polynomial over the y-domain
y_reconstructed = np.linspace(-pitch / 2, pitch / 2, 1000)  # High-resolution y-axis for smooth curve
flux_reconstructed = legendre_poly(y_reconstructed)  # Evaluate the reconstructed flux

# Plot the flux from the mesh tally
plt.figure(figsize=(10, 6))
plt.errorbar(y_bin_centers, mesh_flux/(Δy), yerr=mesh_flux_uncertainty, fmt='o', label='Flux (Mesh Tally)', capsize=3)

# Diagnostics
# import scipy

# int1 = scipy.integrate.trapezoid(mesh_flux/(Δy), y_bin_centers)
# int2 = scipy.integrate.trapezoid(flux_reconstructed, y_reconstructed)
# print(f"Exact integral: {int1} and reconstructed integral {int2}")

# Plot the reconstructed flux from Legendre moments
plt.plot(y_reconstructed, flux_reconstructed, label='Reconstructed Flux (Legendre Moments)', color='red', linewidth=2)

# Add labels, title, and legend
plt.xlabel('y [cm]')
plt.ylabel('Flux')
plt.title('Flux Distribution Along y-Axis at Right Boundary')
plt.grid()
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig('flux_comparison.jpg', dpi=500)

# Show the plot
plt.show()