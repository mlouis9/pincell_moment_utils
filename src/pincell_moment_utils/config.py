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