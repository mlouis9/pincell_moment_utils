from pincell_moment_utils import config
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pincell_moment_utils.postprocessing import SurfaceMeshTally, SurfaceExpansion
import numpy as np

pitch = config.PITCH
ANGULAR_BOUNDS = config.ANGULAR_BOUNDS
SPATIAL_BOUNDS = config.SPATIAL_BOUNDS

def reconstruction_comparison(expansion: SurfaceExpansion, mesh_tally: SurfaceMeshTally, 
                              space_index: int, angle_index: int, energy_index: int,
                              surface: int, option: str='relative_difference'):
    """Compare the surface flux functional reconstruction to the mesh tally from which it was computed
    
    Parameters
    ----------
    expansion
        A surface flux expansion to compare with the mesh tally from which it was created
    mesh_tally
        The mesh tally used to generate the surface flux expansion
    space_index
        The space point (in the mesh defined by the mesh tally) to compute a energy angular slice at
    angle_index
        The angle point (in the mesh defined by the mesh tally) to compute a spatial energy slice at
    energy_index
        The energy point (in the mesh defined by the mesh tally) to compute a spatial angular slice at
    surface
        The surface whose surface fluxes you'd like to compare
    option
        Plot option: 'expansion', 'benchmark', 'relative_difference', either plot the expansion, the benchmark, or the relative difference
        between them
    """
    # Get meshes relevant for plotting and evaluating the reconstructed flux
    space_vals, angle_vals, energy_vals = mesh_tally.meshes[surface]
    energy_filter = mesh_tally.energy_filters[surface]
    E_min = energy_filter.bins[0][0]
    E_max = energy_filter.bins[-1][-1]

    expansion_vals = expansion.evaluate_on_grid(surface, (space_vals, angle_vals, energy_vals))
    mesh_vals = mesh_tally.fluxes[surface]
    vmax = None
    match option:
        case 'relative_difference':
            plot_vals =  np.abs(mesh_vals -expansion_vals)/mesh_vals
            vmax = 1.0
        case 'expansion':
            plot_vals = expansion_vals
        case 'benchmark':
            plot_vals = mesh_vals

    # Plot a 2D slice with energy fixed
    flux_energy_slice = plot_vals[:, :, energy_index]  # Fix energy
    plt.figure(figsize=(8, 6))
    plt.imshow(flux_energy_slice.T, origin="lower", 
               extent=[SPATIAL_BOUNDS[surface][0], SPATIAL_BOUNDS[surface][1], ANGULAR_BOUNDS[surface][0], ANGULAR_BOUNDS[surface][1]], 
               aspect="auto",norm=LogNorm(vmax=vmax))
    plt.colorbar(label="Flux (normalized)")
    plt.xlabel("y-position")
    plt.ylabel("Angle (rad)")
    plt.title(f"Flux Slice at Energy {energy_vals[energy_index]} eV")
    plt.show()

    # Plot a 2D slice with angle fixed
    flux_angle_slice = plot_vals[:, angle_index, :]  # Fix angle
    plt.figure(figsize=(8, 6))
    plt.imshow(flux_angle_slice.T, origin="lower", extent=[SPATIAL_BOUNDS[surface][0], SPATIAL_BOUNDS[surface][1], E_min, E_max], aspect="auto",norm=LogNorm(vmax=vmax))
    plt.colorbar(label="Flux (normalized)")
    plt.ylabel("Energy (eV)")
    plt.xlabel("Position")
    plt.title(f"Flux Slice at Angle {angle_vals[angle_index]} radians")
    plt.show()

    # Plot a 2D slice with y fixed
    flux_y_slice = plot_vals[space_index, :, :]  # Fix y location
    plt.figure(figsize=(8, 6))
    plt.imshow(flux_y_slice.T, origin="lower", extent=[ANGULAR_BOUNDS[surface][0], ANGULAR_BOUNDS[surface][1], E_min, E_max], aspect="auto",norm=LogNorm(vmax=vmax))
    plt.colorbar(label="Flux (normalized)")
    plt.ylabel("Energy (eV)")
    plt.xlabel("Angle (rad)")
    plt.title(f"Flux Slice at Position {space_vals[space_index]}cm")
    plt.show()