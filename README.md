# HZ_evolution

A package to calculate the habitable histories of exoplanets. 


This package aims to constrain the evolutionary histories of planets that we discover. As stars evolve in luminosity and effective temperature, the location of their habitable zones changes in time. This package includes utilities to assess how a planet's habitability will be influenced by its evolving host star. With the `HZ_evolution` package, one can calculate the instellation histories of planets, compute the durations that they spend in the habitable zone as well as the duration they spend interior and exterior to the habitable zone, and calculate the position of the continuously habitable zone among many other features.

If you find this package useful please cite __________

Authors: Noah Tuchow and Jason Wright

## Installation

To install this package clone the github repository to your location of choice:

```bash
git clone https://github.com/nwtuchow/HZ_evolution.git 
``` 

then navigate to the location of `setup.py` and run:

```bash
pip install . 
```

**Note**:  Before using this package, we recommend that users make sure that they have the `isochrones` python package installed. While `isochrones` is a dependency of this package we found that many of the problems with installations come from problems getting the isochrones package working. We recommend following their [installation instructions](https://isochrones.readthedocs.io/en/latest/install.html) and following their [quickstart guide](https://isochrones.readthedocs.io/en/latest/quickstart.html) up until the section on model fitting. The `holoviews` and `Multinest` packages are not used by `HZ_evolution`, so it is the user's choice if they want to install them. Don't worry about running the nosetests recommended by the isochrones package. Some may fail, but this package will still work.

The first time one runs `isochrones` it will take a while as it needs to download and cache the MIST stellar model grid.

## Dependencies

In order to use the `HZ_evolution` package one requires a user supplied stellar model. By default it includes utilities to use the MIST (MESA Isochrones and Stellar Tracks) model grid via `isochrones.py`, but any stellar model can be used.

It also requires the user to specify a formulation for the habitable zone. Many common formulations are included in `HZ_evolution`, and one has the option to define their own formulation of the habitable zone.

`HZ_evolution` requires the following python packages:

- numpy,
- matplotlib
- pandas
- numba
- nose
- pytables
- astropy
- scipy 
- emcee
- isochrones
- corner

These packages should all be available via pip



## Getting Started

For beginners we recommend starting with the examples in the notebooks/ directory. Start first with `Intro_to_HZ_evolution.ipynb`.

More information about the usage of the functions can be found in the docstrings of the functions.

Full documentation will be added in the future.
