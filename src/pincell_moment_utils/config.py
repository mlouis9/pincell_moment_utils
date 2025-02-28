import numpy as np
"""A package level configuration module."""
PITCH = 1.26
"""Lattice pitch of pincell in cm"""
TRANSFORM_FUNCTIONS = [
    lambda x:  np.sin(x),
    lambda x: -np.sin(x),
    lambda x: -np.cos(x),
    lambda x:  np.cos(x)
]
"""Functions used for transforming the outgoing angular domain to that of the Legendre polynomials for each surface in order 1,2,3,4. 
It is not recommended to change these."""
WEIGHT_FUNCTIONS = [
    lambda x:  np.cos(x),
    lambda x: -np.cos(x),
    lambda x:  np.sin(x),
    lambda x: -np.sin(x)
]
"""Weight functions in the angular variable used when computing moments of the functional expansion. It is not recommended to change these."""
OUTGOING_ANGULAR_BOUNDS = [
    [-np.pi/2, np.pi/2],
    [np.pi/2, 3/2*np.pi],
    [0, np.pi],
    [-np.pi, 0]
]
"""Angular bounds for outgoing direction on each surface."""
SPATIAL_BOUNDS = [
    [-PITCH/2, PITCH/2],
    [-PITCH/2, PITCH/2],
    [-PITCH/2, PITCH/2],
    [-PITCH/2, PITCH/2],
]
"""Spatial bounds that limit surface extent. It's not clear that this would ever (or could ever) be changed without making large changes
to the code, but it is included for consistency with the angular bounds."""
INCIDENT_OUTGOING_PERMUTATION = [
    1, 0, 3, 2
]
"""Permutation which maps the incoming angular bounds to the outgoing angular bounds."""
SURFACE_PERPENDICULAR_COORDINATE = [PITCH/2, -PITCH/2, PITCH/2, -PITCH/2]
SURFACE_COORD_TO_3D = [ 
    lambda x: (SURFACE_PERPENDICULAR_COORDINATE[0], x, 0),
    lambda x: (SURFACE_PERPENDICULAR_COORDINATE[1], x, 0),
    lambda x: (x, SURFACE_PERPENDICULAR_COORDINATE[2], 0),
    lambda x: (x, SURFACE_PERPENDICULAR_COORDINATE[3], 0)
]
"""Functions which map the surface perpendicular coordinate to a 3D coordinate. This is used when generating the source file for OpenMC."""
INCIDENT_ANGLE_TRANSFORMATIONS = [
    [lambda x: x          , lambda x: x - np.pi   , lambda x: x + np.pi/2  , lambda x: x - np.pi/2],
    [lambda x: x + np.pi  , lambda x: x           , lambda x: x + 3/2*np.pi, lambda x: x + np.pi/2],
    [lambda x: x - np.pi/2, lambda x: x -3/2*np.pi, lambda x: x            , lambda x: x - np.pi  ],
    [lambda x: x + np.pi/2, lambda x: x - np.pi/2 , lambda x: x + np.pi    , lambda x: x          ]
]
"""An array of functinos that transform (incident) angles from a surface i to those of a surface j by application of the function
INCIDENT_ANGLE_TRANSFORMATIONS[i][j]. This is used when generating source files for OpenMC."""