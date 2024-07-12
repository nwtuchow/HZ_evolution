# HZ_Evolution

A package to calculate the habitable histories of exoplanets. 


This package aims to constrain the evolutionary histories of planets that we discover. As stars evolve in luminosity and effective temperature, the location of their habitable zones changes in time. This package includes utilities to assess how a planet's habitability will be influenced by its evolving host star. With the `HZ_evolution` package, one can calculate the instellation histories of planets, compute the durations that they spend in the habitable zone as well as the duration they spend interior and exterior to the habitable zone, and calculate the position of the continuously habitable zone among many other features.

If you find this package useful please cite __________

Authors: Noah Tuchow and Jason Wright

##Installation

To install this package clone the github repository to your location of choice:

```bash
git clone https://github.com/nwtuchow/HZ_evolution.git 
``` 

then navigate to the location of `setup.py` and run:

```bash
pip install . 
```

##Dependencies

In order to use the `HZ_evolution` package one requires a user supplied stellar model. By default it includes utilities to use the MIST (MESA Isochrones and Stellar Tracks) model grid via `isochrones.py`, but any stellar model can be used.

It also requires the user to specify a formulation for the habitable zone. Many common formulations are included in `HZ_evolution`, and one has the option to define their own formulation of the habitable zone.

`HZ_evolution` requires the following python packages:

- numpy,
- matplotlib
- pandas
- astropy
- scipy 
- emcee
- isochrones
- corner

These packages should all be available via pip

**Note**: We recommend testing the `isochrones` python package to make sure it is working prior to running `HZ_evolution`. The first time one runs `isochrones` it may need to cache a stellar model grid, so it might take a little while.

##Getting Started

For beginners we recommend starting with the examples in the notebooks/ directory. Start first with `Intro_to_HZ_evolution.ipynb`.

More information about the usage of the functions can be found in the docstrings of the functions.

Full documentation will be added in the future.