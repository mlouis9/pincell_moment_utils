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

# Set vacuum boundary condition (an incident flux source will be added later)
box = openmc.model.rectangular_prism(width=pitch, height=pitch,
                               boundary_type='vacuum')
water_region = box & +clad_or

moderator = openmc.Cell(4, 'moderator')
moderator.fill = water
moderator.region = water_region

root = openmc.Universe(cells=(fuel, gap, clad, moderator))

geom = openmc.Geometry()
geom.root_universe = root
geom.export_to_xml()

root.plot(width=(1.3, 1.3), pixels=int(5E+05))
plt.savefig('geometry.png', dpi=500)

#==========
# Settings
#==========
# Define incident flux source
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


settings = openmc.Settings()
settings.source = [src_px, src_mx, src_py, src_my]
# point = openmc.stats.Point((0.0, 0.0, 0.0))
# settings.source = openmc.source.IndependentSource(space=point)
settings.batches = 100
settings.inactive = 20
settings.particles = 10000
settings.run_mode = 'fixed source'
settings.export_to_xml()

# ==========
# Tallies
# ==========
tallies = openmc.Tallies()

# Create a mesh for tallying
mesh = openmc.RegularMesh()
N = 15
mesh.dimension = [N, N, 1]  # 100x100 mesh in XY plane, 1 bin in Z
mesh.lower_left = [-pitch / 2, -pitch / 2, -1e-5]
mesh.upper_right = [pitch / 2, pitch / 2, 1e-5]

# Create a mesh filter
mesh_filter = openmc.MeshFilter(mesh)

# Flux tally
flux_tally = openmc.Tally(name='flux')
flux_tally.filters = [mesh_filter]
flux_tally.scores = ['flux']
tallies.append(flux_tally)

# Fission density tally
fission_tally = openmc.Tally(name='fission')
fission_tally.filters = [mesh_filter]
fission_tally.scores = ['fission']
tallies.append(fission_tally)

# Export tallies to XML
tallies.export_to_xml()


openmc.run(threads=8)