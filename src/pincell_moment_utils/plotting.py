from pincell_moment_utils import config
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from pincell_moment_utils.postprocessing import SurfaceMeshTally, SurfaceExpansionBase
import numpy as np

pitch = config.PITCH
ANGULAR_BOUNDS = config.OUTGOING_ANGULAR_BOUNDS
SPATIAL_BOUNDS = config.SPATIAL_BOUNDS

def plot_expansion(expansion: SurfaceExpansionBase, space_index: int, angle_index: int, energy_index: int, surface: int, 
                   N_space, N_angular, incident: bool=False):
    """Plot the surface flux functional expansion

    Parameters
    ----------
    expansion
        A surface flux expansion to plot
    space_index
        The space point (in the mesh defined by the mesh tally) to compute a energy angular slice at
    angle_index
        The angle point (in the mesh defined by the mesh tally) to compute a spatial energy slice at
    energy_index
        The energy point (in the mesh defined by the mesh tally) to compute a spatial angular slice at
    surface
        The surface whose surface fluxes you'd like to plot
    N_space
        The number of spatial points to plot
    N_angular
        The number of angular points to plot
    incident
        If True, the incident fluxes are plotted instead of the outgoing fluxes.
    """

    space_vals = np.linspace(SPATIAL_BOUNDS[surface][0], SPATIAL_BOUNDS[surface][1], N_space)
    if incident:
        permutation = config.INCIDENT_OUTGOING_PERMUTATION
    else:
        permutation = np.arange(4)
    angle_vals = np.linspace(ANGULAR_BOUNDS[permutation[surface]][0], ANGULAR_BOUNDS[permutation[surface]][1], N_angular)
    energy_vals = np.diff(expansion.energy_filters[surface].values)
    E_min = expansion.energy_filters[surface].bins[0][0]
    E_max = expansion.energy_filters[surface].bins[-1][-1]
    expansion_vals = expansion.evaluate_on_grid(surface, (space_vals, angle_vals, energy_vals))
    vmax = np.max(expansion_vals)
    vmin = np.min(expansion_vals)
    if vmax > 0 and vmin > 0:
        norm = LogNorm(vmax=vmax)
    elif vmax == 0:
        norm = Normalize(vmin=vmin, vmax=1E-10)  # Linear scale with small max
    else:
        norm = Normalize(vmin=0, vmax=vmax)

    # Plot a 2D slice with energy fixed
    flux_energy_slice = expansion_vals[:, :, energy_index]  # Fix energy
    plt.figure(figsize=(8, 6))
    plt.imshow(flux_energy_slice.T, origin="lower", 
               extent=[SPATIAL_BOUNDS[surface][0], SPATIAL_BOUNDS[permutation[surface]][1], ANGULAR_BOUNDS[permutation[surface]][0], ANGULAR_BOUNDS[permutation[surface]][1]], 
               aspect="auto",norm=norm)
    plt.colorbar(label="Flux (normalized)")
    plt.xlabel("Position")
    plt.ylabel("Angle (rad)")
    plt.title(f"Flux Slice at Energy {energy_vals[energy_index]:1.4E} eV")
    plt.show()

    # Plot a 2D slice with angle fixed
    flux_angle_slice = expansion_vals[:, angle_index, :]  # Fix angle
    plt.figure(figsize=(8, 6))
    plt.imshow(flux_angle_slice.T, origin="lower", extent=[SPATIAL_BOUNDS[surface][0], SPATIAL_BOUNDS[surface][1], E_min, E_max], aspect="auto",norm=norm)
    plt.colorbar(label="Flux (normalized)")
    plt.ylabel("Energy (eV)")
    plt.xlabel("Position")
    plt.title(f"Flux Slice at Angle {angle_vals[angle_index]:1.4f} radians")
    plt.show()

    # Plot a 2D slice with y fixed
    flux_y_slice = expansion_vals[space_index, :, :]  # Fix y location
    plt.figure(figsize=(8, 6))
    plt.imshow(flux_y_slice.T, origin="lower", extent=[ANGULAR_BOUNDS[permutation[surface]][0], ANGULAR_BOUNDS[permutation[surface]][1], E_min, E_max], aspect="auto",norm=norm)
    plt.colorbar(label="Flux (normalized)")
    plt.ylabel("Energy (eV)")
    plt.xlabel("Angle (rad)")
    plt.title(f"Flux Slice at Position {space_vals[space_index]:1.4f} cm")
    plt.show()

def plot_expansion_slice(expansion: SurfaceExpansionBase, energy_index: int, surface: int, 
                         N_space: int=40, N_angular: int=20, n_step: int=4, incident: bool=False):
    """Plot the 1D slices of the surface flux with respect to space and angle at a fixed energy index.
    
    Parameters
    ----------
    expansion
        A surface flux expansion to plot
    energy_index
        The energy point (in the mesh defined by the mesh tally) to compute a spatial angular slice at
    surface
        The surface whose surface fluxes you'd like to plot
    N_space
        The number of spatial points to plot
    N_angular
        The number of angular points to plot
    incident
        If True, the incident fluxes are plotted instead of the outgoing fluxes.
    """
    space_vals = np.linspace(SPATIAL_BOUNDS[surface][0], SPATIAL_BOUNDS[surface][1], N_space)
    if incident:
        permutation = config.INCIDENT_OUTGOING_PERMUTATION
    else:
        permutation = np.arange(4)
    angle_vals = np.linspace(ANGULAR_BOUNDS[permutation[surface]][0], ANGULAR_BOUNDS[permutation[surface]][1], N_angular)
    energy_vals = np.diff(expansion.energy_filters[surface].values)
    expansion_vals = expansion.evaluate_on_grid(surface, (space_vals, angle_vals, energy_vals))

    n_colors = len(range(0, len(angle_vals), n_step))  # Ensure we generate enough colors
    colors = plt.cm.rainbow(np.linspace(0, 1, n_colors))  # Match number of colors to loop steps

    for i, index in enumerate(range(0, len(angle_vals), n_step)):
        plt.plot(space_vals, expansion_vals[:, index, energy_index], color=colors[i], linestyle='dashed', label=f'expansion ω={angle_vals[index]:1.2f}')

    plt.grid()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Position (cm)')
    plt.ylabel('Flux')
    plt.show()

    n_colors = len(range(0, len(space_vals), n_step))  # Ensure we generate enough colors
    colors = plt.cm.rainbow(np.linspace(0, 1, n_colors))  # Match number of colors to loop steps

    for i, index in enumerate(range(0, len(space_vals), n_step)):
        plt.plot(angle_vals, expansion_vals[index, :, energy_index], color=colors[i], linestyle='dashed', label=f'expansion y={space_vals[index]:1.2f}')

    plt.grid()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel('Angle (radians)')
    plt.ylabel('Flux')
    plt.show()

    plt.plot(space_vals, np.sum(expansion_vals, (1,2)), linestyle='dashed', color='tab:blue', label='expansion')
    plt.xlabel('Position (cm)')
    plt.ylabel('"Integrated" Flux')
    plt.grid()
    plt.legend()
    plt.show()


def reconstruction_comparison(expansion: SurfaceExpansionBase, mesh_tally: SurfaceMeshTally, 
                              space_index: int, angle_index: int, energy_index: int,
                              surface: int, option: str='relative_difference', incident: bool=False):
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
        Plot option: 'expansion', 'mesh_tally', 'relative_difference', either plot the expansion, the mesh_tally, or the relative difference
        between them
    incident
        If True, the incident fluxes are plotted instead of the outgoing fluxes.
    """

    if incident:
        permutation = config.INCIDENT_OUTGOING_PERMUTATION
        surface = permutation[surface]
        

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
        case 'mesh_tally':
            plot_vals = mesh_vals
        case _:
            raise ValueError(f"Invalid option: {option}. Choose from 'relative_difference', 'expansion', or 'mesh_tally'.")

    # Plot a 2D slice with energy fixed
    flux_energy_slice = plot_vals[:, :, energy_index]  # Fix energy
    plt.figure(figsize=(8, 6))
    plt.imshow(flux_energy_slice.T, origin="lower", 
               extent=[SPATIAL_BOUNDS[surface][0], SPATIAL_BOUNDS[surface][1], ANGULAR_BOUNDS[surface][0], ANGULAR_BOUNDS[surface][1]], 
               aspect="auto",norm=LogNorm(vmax=vmax))
    plt.colorbar(label="Flux (normalized)")
    plt.xlabel("Position")
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