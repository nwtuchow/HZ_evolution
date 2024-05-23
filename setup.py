import os
from setuptools import setup

setup(
      name='HZ_evolution',
      version='0.0.1',
      author='Noah Tuchow',
      description="A package to calculate the habitable histories of exoplanets",
      install_requires=[
          'numpy',
          'matplotlib',
          'pandas',
          'astropy',
          'scipy',
          'emcee',
          'isochrones'],
      packages=['HZ_evolution'],
      include_package_data=True
      )