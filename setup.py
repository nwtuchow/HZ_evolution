import os
from setuptools import setup

setup(
      name='HZ_evolution',
      version='1.0.0',
      author='Noah Tuchow',
      description="A package to calculate the habitable histories of exoplanets",
      install_requires=[
          'numpy',
          'matplotlib',
          'pandas',
          'numba',
          'nose',
          'astropy',
          'scipy',
          'emcee',
          'isochrones',
          'corner'],
      packages=['HZ_evolution']
      )
