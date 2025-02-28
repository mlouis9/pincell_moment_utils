import openmc
import numpy as np
import h5py
import argparse
from pathlib import Path
import openmc.lib
import tempfile
import os

# Parsing command line argument for source file
parser = argparse.ArgumentParser()
parser.add_argument("source_file", type=str, help="Absolute path to OpenMC source file for incident flux.")
args = parser.parse_args()
source_file = Path(args.source_file).resolve()

# Create a temporary directory where we will place all intermediate files
with tempfile.TemporaryDirectory() as tempdir:
    #===========
    # Materials
    #===========

    uo2 = openmc.Material(name='fuel')
    uo2.add_nuclide('U235', $wt_enrichment)
    uo2.add_nuclide('U238', 1-$wt_enrichment)
    uo2.add_nuclide('O16', 2.0)
    uo2.set_density('g/cm3', $fuel_density)

    zirconium = openmc.Material(2, "zirconium")
    zirconium.add_element('Zr', 1.0)
    zirconium.set_density('g/cm3', 6.6)

    water = openmc.Material(3, "h2o")
    water.add_nuclide('H1', 2.0)
    water.add_nuclide('O16', 1.0)
    water.set_density('g/cm3', $water_density)
    water.add_s_alpha_beta('c_H_in_H2O')

    mats = openmc.Materials([uo2, zirconium, water])
    mats.export_to_xml(Path(tempdir) / 'materials.xml')

    #==========
    # Geometry
    #==========

    # Create shapes
    fuel_or = openmc.ZCylinder(r=0.39)
    clad_ir = openmc.ZCylinder(r=0.40)
    clad_or = openmc.ZCylinder(r=0.46)

    # Create regions
    fuel_region = -fuel_or
    gap_region = +fuel_or & -clad_ir
    clad_region = +clad_ir & -clad_or

    # Create cells
    fuel = openmc.Cell(1, 'fuel')
    fuel.fill = uo2
    fuel.region = fuel_region

    gap = openmc.Cell(2, 'air gap')
    gap.region = gap_region

    clad = openmc.Cell(3, 'clad')
    clad.fill = zirconium
    clad.region = clad_region

    pitch = $pitch

    left   = openmc.XPlane(x0=-pitch/2)
    right  = openmc.XPlane(x0=pitch/2)
    bottom = openmc.YPlane(y0=-pitch/2)
    top    = openmc.YPlane(y0=pitch/2)

    water_region = +left & -right & +bottom & -top & +clad_or

    # Define the moderator
    moderator = openmc.Cell(4, 'moderator')
    moderator.fill = water
    moderator.region = water_region

    # --------------
    # Tally Regions
    # --------------
    box = openmc.model.rectangular_prism(width=3/2*pitch, height=3/2*pitch,
                                boundary_type='vacuum')

    # Define artificial (vacuum) regions for tallying angular flux
    vacuum_region       = box & ~water_region
    right_tally_region  = box & +right  & -top  & +bottom
    left_tally_region   = box & -left   & -top  & +bottom
    top_tally_region    = box & +top    & +left & -right
    bottom_tally_region = box & -bottom & +left & -right

    # Define junk regions which are nonetheless neessary for fully defining the geometry
    top_right    = box & +right & +top
    top_left     = box & -left  & +top
    bottom_right = box & +right & -bottom
    bottom_left  = box & -left  & -bottom

    # ------------
    # Tally Cells
    # ------------
    # Tallies

    right_tally_cell = openmc.Cell(11, 'right_tally')
    right_tally_cell.region = right_tally_region

    left_tally_cell = openmc.Cell(12, 'left_tally')
    left_tally_cell.region = left_tally_region

    top_tally_cell = openmc.Cell(13, 'top_tally')
    top_tally_cell.region = top_tally_region

    bottom_tally_cell = openmc.Cell(14, 'bottom_tally')
    bottom_tally_cell.region = bottom_tally_region

    # Junk cells
    top_right_cell           = openmc.Cell(15, 'top_right')
    top_right_cell.region    = top_right
    top_left_cell            = openmc.Cell(16, 'top_left')
    top_left_cell.region     = top_left
    bottom_left_cell         = openmc.Cell(17, 'bottom_left')
    bottom_left_cell.region  = bottom_left
    bottom_right_cell        = openmc.Cell(18, 'bottom_right')
    bottom_right_cell.region = bottom_right

    root = openmc.Universe(cells=(fuel, gap, clad, moderator, 
                                right_tally_cell, left_tally_cell, top_tally_cell, bottom_tally_cell, 
                                top_right_cell, top_left_cell, bottom_left_cell, bottom_right_cell))

    geom = openmc.Geometry()
    geom.root_universe = root
    geom.export_to_xml(Path(tempdir) / 'geometry.xml')

    #==========
    # Settings
    #==========

    # ------------------
    # Particle Settings
    # ------------------
    settings = openmc.Settings()
    settings.source = openmc.FileSource(str(source_file))
    settings.batches = $num_batches
    settings.inactive = $num_inactive
    settings.particles = $num_particles_per_generation
    settings.run_mode = 'fixed source'
    settings.export_to_xml(Path(tempdir) / 'settings.xml')

    # ========
    # Tallies
    # ========
    tallies = openmc.Tallies()

    # Cell filters
    # ^^^^^^^^^^^^
    tally_cell_ids = [right_tally_cell.id, left_tally_cell.id, top_tally_cell.id, bottom_tally_cell.id]
    surface_filters = [openmc.CellFilter([tally_cell_id]) for tally_cell_id in tally_cell_ids]
    fuel_filter = openmc.CellFilter([fuel])

    # Zernlike expansion filter
    # ^^^^^^^^^^^^^^^^^^^^^^^^^
    order = $zernlike_order
    radius = 0.39 # Fuel OR could make it larger in case we want some flux information near the fuel pin, but not for now
    flux_tally_zernike = openmc.Tally(name = "zernike")
    flux_tally_zernike.id = 5
    flux_tally_zernike.scores = ['flux']
    zernike_filter = openmc.ZernikeFilter(order=order, x=0.0, y=0.0, r=radius)
    flux_tally_zernike.filters = [fuel_filter, zernike_filter]
    tallies.append(flux_tally_zernike)

    # Meshes along surfaces
    # ^^^^^^^^^^^^^^^^^^^^^
    mesh_filters = []
    N = $N # number of mesh points along each surface
    dimensions = [[1, N], [1, N], [N, 1], [N, 1]]
    lower_lefts = [[pitch/2, -pitch/2], [-3/4*pitch, -pitch/2], [-pitch/2, pitch/2], [-pitch/2, -3/4*pitch]]
    upper_rights = [[3/4*pitch, pitch/2], [-pitch/2, pitch/2], [pitch/2, 3/4*pitch], [pitch/2, -pitch/2]]
    for surface in range(4):
        mesh = openmc.Mesh()
        mesh.dimension = dimensions[surface]  # 1 bin in x, 20 bins in y
        mesh.lower_left = lower_lefts[surface]  # Narrow region near the right boundary
        mesh.upper_right = upper_rights[surface]  # Extend over the y range
        mesh_filter = openmc.MeshFilter(mesh)
        mesh_filters.append(mesh_filter)

    # Energy filter for multigroup energy binning
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Read in CASMO 70 group structure from h5
    with h5py.File('$energy_file', 'r') as f:
        energy_groups = f['energy groups'][:]

    energy_filter = openmc.EnergyFilter(energy_groups)

    # Filters for angular binning on surfaces
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Nω = $N_omega
    angle_filters = []

    # Special handling of branch cut of angular domain
    minus_x_angles = np.concatenate(
        (np.linspace(np.pi/2, np.pi, Nω//2 + 1),
        np.linspace(-np.pi, -np.pi/2, Nω//2 + 1))
    )
    minus_x_angles = np.sort(minus_x_angles) 
    # Angles no longer in descending order, and creates a artifactual bin at [-π/2, π/2] that must be handled in postprocessing, but
    # required for monotonicity of the angular mesh required by OpenMC

    angle_ranges_out = [np.linspace(-np.pi/2, np.pi/2, Nω+1), 
                        minus_x_angles,
                        np.linspace(0, np.pi, Nω+1),
                        np.linspace(-np.pi, 0, Nω+1)] # ω ranges corresponding to the outgoig direction on each surface
    for surface in range(4):
        angle_range = angle_ranges_out[surface]
        filter = openmc.AzimuthalFilter(angle_ranges_out[surface])
        angle_filters.append(filter)


    # Outgoing flux tallies on surfaces
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    tally_names = [f'flux_at_{side}_boundary' for side in ['right', 'left', 'top', 'bottom']]

    for surface in range(4):
        tally = openmc.Tally(name = tally_names[surface])
        tally.id = surface+1
        tally.filters = [surface_filters[surface], mesh_filters[surface], angle_filters[surface], energy_filter]
        tally.scores = ['flux']
        tallies.append(tally)

    # keff tally
    # ^^^^^^^^^^
    # Tally for counting fissions in the fuel
    fission_tally = openmc.Tally(name='keff')
    fission_tally.id = 6
    fuel_filter = openmc.CellFilter(fuel.id)
    fission_tally.filters = [fuel_filter]
    fission_tally.scores = ['fission']
    tallies.append(fission_tally)

    # Export tallies to XML
    tallies.export_to_xml(Path(tempdir) / 'tallies.xml')


    # Extract source_name from source_file argument
    source_name = Path(args.source_file).stem  # Extracts filename without extension

    # Run OpenMC in library mode to control statepoint naming
    os.chdir(tempdir)
    openmc.lib.init()  # Initialize OpenMC in memory
    openmc.lib.run()   # Run the simulation
    os.chdir(Path(__file__).parent)

    # Manually write statepoint with custom filename
    custom_statepoint_name = f"statepoint.{source_name}.h5"
    openmc.lib.statepoint_write(custom_statepoint_name)

    # Finalize OpenMC
    openmc.lib.finalize()