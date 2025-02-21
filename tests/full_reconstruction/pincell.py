import openmc
import matplotlib.pyplot as plt
import numpy as np
import h5py

#===========
# Materials
#===========

uo2 = openmc.Material(name='fuel')
uo2.add_nuclide('U235', 0.23)
uo2.add_nuclide('U238', 0.77)
uo2.add_nuclide('O16', 2.0)
uo2.set_density('g/cm3', 10.0)

zirconium = openmc.Material(2, "zirconium")
zirconium.add_element('Zr', 1.0)
zirconium.set_density('g/cm3', 6.6)

water = openmc.Material(3, "h2o")
water.add_nuclide('H1', 2.0)
water.add_nuclide('O16', 1.0)
water.set_density('g/cm3', 1.0)
water.add_s_alpha_beta('c_H_in_H2O')

# Define a neutronically irrelevant material
dummy_material = openmc.Material(name='dummy')
dummy_material.add_nuclide('Ar40', 1.0)
dummy_material.set_density('g/cm3', 1)

mats = openmc.Materials([uo2, zirconium, water, dummy_material])
mats.export_to_xml()

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

pitch = 1.26

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
right_tally_cell.fill = dummy_material

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
geom.export_to_xml()

root.plot(width=(3/2*pitch, 3/2*pitch), pixels=int(5E+05), legend=True, colors={
    moderator: 'blue',
    top_right_cell: 'grey',
    top_left_cell: 'grey',
    bottom_left_cell: 'grey',
    bottom_right_cell: 'grey',
    right_tally_cell: 'green',
    left_tally_cell: 'dodgerblue',
    top_tally_cell: 'orange',
    bottom_tally_cell: 'red',
    fuel: 'yellow',
    clad: 'grey',
    gap: 'white'
})
plt.tight_layout()
plt.savefig('geometry.png', dpi=500)

#==========
# Settings
#==========

# ---------------------
# Incident Flux Source
# ---------------------
# Currently defining via openmc stats distributions, but will be supplied by a file in the final version
boundary_offset = 1e-4  # Slight inward shift to avoid exact boundaries

# Adjust px, mx, py, my distributions
px_dist = openmc.stats.Discrete([pitch/2 - boundary_offset], [1.0])
mx_dist = openmc.stats.Discrete([-pitch/2 + boundary_offset], [1.0])
py_dist = openmc.stats.Discrete([pitch/2 - boundary_offset], [1.0])
my_dist = openmc.stats.Discrete([-pitch/2 + boundary_offset], [1.0])

x_dist = openmc.stats.Normal(0, 0.3)
y_dist = openmc.stats.Normal(0, 0.3)
z_dist = openmc.stats.Discrete([0.0], [1.0])  # z fixed at 0

energy_dist = openmc.stats.Discrete([0.0253], [1.0])

angle_distpx = openmc.stats.Monodirectional((1.0, 0.0, 0.0))
angle_distmx = openmc.stats.Monodirectional((-1.0, 0.0, 0.0))
angle_distpy = openmc.stats.Monodirectional((0.0, 1.0, 0.0))
angle_distmy = openmc.stats.Monodirectional((0.0, -1.0, 0.0))

spatial_distpx = openmc.stats.CartesianIndependent(px_dist, y_dist, z_dist)
spatial_distmx = openmc.stats.CartesianIndependent(mx_dist, y_dist, z_dist)
spatial_distpy = openmc.stats.CartesianIndependent(x_dist, py_dist, z_dist)
spatial_distmy = openmc.stats.CartesianIndependent(x_dist, my_dist, z_dist)
src_px = openmc.Source(space=spatial_distpx, angle=angle_distmx, energy=energy_dist)
src_mx = openmc.Source(space=spatial_distmx, angle=angle_distpx, energy=energy_dist)
src_py = openmc.Source(space=spatial_distpy, angle=angle_distmy, energy=energy_dist)
src_my = openmc.Source(space=spatial_distmy, angle=angle_distpy, energy=energy_dist)

# ------------------
# Particle Settings
# ------------------
settings = openmc.Settings()
settings.source = [src_px, src_mx, src_py, src_my]
settings.batches = 100
settings.inactive = 10
settings.particles = 10000
settings.run_mode = 'fixed source'
settings.export_to_xml()

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
order = 4
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
N = 40 # number of mesh points along each surface
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
with h5py.File('../data/cas8.h5', 'r') as f:
    energy_groups = f['energy groups'][:]

energy_filter = openmc.EnergyFilter(energy_groups)

# Filters for angular binning on surfaces
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Nω = 20
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
tallies.export_to_xml()


openmc.run()