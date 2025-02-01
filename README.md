# Introduction
This project primarily serves as a reference to the methods used to generate a moment dataset for parameterizing the incident and outgoing fluxes, as well as the local multiplication factor and the pin flux distribution for training a machine learning model to implement the incident flux response method. Included is also a utility package which was used to postprocess moment and mesh tallies and perform flux reconstruction from these moments following the theory described in `TODO`.

# Installation and Setup
This package is [published on PyPI](https://pypi.org/project/pincell_moment_utils/), and so can be installed (along with all of the necessary dependencies) via `pip`
```
pip install pincell_moment_utils
```
Note some of the features require the ability to run transport simulations with OpenMC, which require a valid set of cross sections, which can be installed using the scripts [here](https://github.com/openmc-dev/data).