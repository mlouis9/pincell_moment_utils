import openmc
import matplotlib.pyplot as plt

#===========
# Materials
#===========

uo2 = openmc.Material(name='fuel')
uo2.add_nuclide('U235', 0.03)
uo2.add_nuclide('U238', 0.97)
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
mats = openmc.Materials([uo2, zirconium, water])
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

root.plot(width=(1.3, 1.3), pixels=int(5E+05))
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

x_dist = openmc.stats.Uniform(-pitch/2, pitch/2)
y_dist = openmc.stats.Uniform(-pitch/2, pitch/2)
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
# point = openmc.stats.Point((0.0, 0.0, 0.0))
# settings.source = openmc.source.IndependentSource(space=point)
settings.batches = 100
settings.inactive = 20
settings.particles = 10000
settings.run_mode = 'fixed source'
settings.export_to_xml()

# ========
# Tallies
# ========
tallies = openmc.Tallies()

right_filter = openmc.CellFilter([right_tally_cell.id])
left_filter = openmc.CellFilter([left_tally_cell.id])
top_filter = openmc.CellFilter([top_tally_cell.id])
bottom_filter = openmc.CellFilter([bottom_tally_cell.id])

rightflux_tally = openmc.Tally(name = 'flux_at_right_boundary')
rightflux_tally.filters = [right_filter]
rightflux_tally.scores = ['flux']

# Export tallies to XML
tallies.append(rightflux_tally)
tallies.export_to_xml()


openmc.run(threads=8)