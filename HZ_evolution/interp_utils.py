#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities to create an interpolator for habitable durations.
Includes 3D and 4D versions of the interpolator with and without Fe/H as a dimension
These functions use the output of calculate_grid_values.py and calculate_grid_values_4D.py
so run these scripts beforehand or these functions won't work'
"""

import pandas as pd
from isochrones.interp import DFInterpolator
import os
from .hz_utils import ROOT_DIR, OUTPUT_DIR

def calculate_grid():
    '''calculates habitable durations for the grid of MIST models at solar metallicity
    run this function before calling construct_interpolator_3D
    time consuming and memory intensive, benmarked at 824s'''
    import calculate_grid_values
    
def calculate_grid_4D():
    '''calculates habitable durations for the grid of MIST models, including metallicity as a dimension
    run this function before calling construct_interpolator_4d
    time consuming and memory intensive, benchmarked at 5054s'''
    import calculate_grid_values_4D



def construct_interpolator_3D(mass_min= 0.1,mass_max= 2.0, eep_min=6,eep_max=605,
                              Seff_min=0.1,Seff_max=2.1,
                              fname=OUTPUT_DIR+"tau_df_K13_optimistic.csv"):
    '''
    Creates a 3D interpolator for habitable duration, time interior to the HZ and 
    time exterior to the HZ  at [Fe/H]=0.
    Uses the isochrones.py DFInterpolator object.
    Resulting interpolator will take an array of parameters as an argument 
    with the following parameters:
        pars= ['initial_mass','EEP','S_eff']

    Specify min and max for each dimension to construct an interpolator from a 
    subset of the model grid to be faster and use less RAM.
    
    Parameters
    ----------
    mass_min : float, optional
        min mass. The default is 0.1.
    mass_max : float, optional
        max mass. The default is 2.0.
    eep_min : float, optional
        min eep. The default is 6.
    eep_max : float, optional
        max eep. The default is 605.
    Seff_min : float, optional
        min Seff. The default is 0.1.
    Seff_max : float, optional
        max Seff. The default is 2.1.
    fname : TYPE, optional
        File name for model grid. The default is the output file from calculate_grid_3D.

    Returns
    -------
    tau_interp : DFInterpolator
        Interpolator for the grid of tau, t_int, and t_ext

    '''
    
    if not os.path.exists(fname):
        print("Error: File '%s' not found. If you haven't run calculate_grid, do so before using this function" % fname)
        return
    
    index_cols=['initial_mass','EEP','S_eff']
    all_cols=['initial_mass', 'EEP', 'S_eff', 'tau', 't_int','t_ext']
    tau_chunks=pd.read_csv(fname, chunksize=10**4)
    tau_df= pd.DataFrame(columns=all_cols)
    for chunk in tau_chunks:
        temp_df=chunk.loc[(chunk['initial_mass']>=mass_min) & (chunk['initial_mass']<=mass_max) &
                          (chunk['EEP']>=eep_min) & (chunk['EEP']<=eep_max) &
                          (chunk['S_eff']>=Seff_min) & (chunk['S_eff']<=Seff_max)]
        if len(temp_df)>0:
            if len(tau_df)==0:
                tau_df=temp_df
            else:
                tau_df=pd.concat([tau_df,temp_df])
    
    tau_df=tau_df.set_index(index_cols)

    tau_interp= DFInterpolator(tau_df)
    return tau_interp


def construct_interpolator_4D(feh_min=-2.0,feh_max=0.5,mass_min= 0.1,mass_max= 2.0,
                           eep_min=6,eep_max=605,Seff_min=0.1,Seff_max=2.1,
                           fname=OUTPUT_DIR+"tau_df_K13_optimistic_4D.csv"):
    '''
    Creates a 4D interpolator for habitable duration, time interior to the HZ and 
    time exterior to the HZ using the isochrones.py DFInterpolator object.
    Resulting interpolator will take an array of parameters as an argument 
    with the following parameters:
        pars= ['initial_Fe_H','initial_mass','EEP','S_eff']

    Specify min and max for each dimension to construct an interpolator from a 
    subset of the model grid to be faster and use less RAM.
    
    Parameters
    ----------
    feh_min : float, optional
        min [Fe/H] . The default is -2.0.
    feh_max : float, optional
        max [Fe/H]. The default is 0.5.
    mass_min : float, optional
        min mass. The default is 0.1.
    mass_max : float, optional
        max mass. The default is 2.0.
    eep_min : float, optional
        min eep. The default is 6.
    eep_max : float, optional
        max eep. The default is 605.
    Seff_min : float, optional
        min Seff. The default is 0.1.
    Seff_max : float, optional
        max Seff. The default is 2.1.
    fname : TYPE, optional
        File name for model grid. The default is the output file from calculate_grid_4D.

    Returns
    -------
    tau_interp : DFInterpolator
        Interpolator for the grid of tau, t_int, and t_ext

    '''
    if not os.path.exists(fname):
        print("Error: File '%s' not found. If you haven't run calculate_grid_4D do so before using this function" % fname)
        return
        
    
    index_cols=['initial_Fe_H','initial_mass','EEP','S_eff']
    all_cols=['initial_Fe_H', 'initial_mass', 'EEP', 'S_eff', 'tau', 't_int','t_ext']
    tau_chunks=pd.read_csv(fname, chunksize=10**4)
    tau_df= pd.DataFrame(columns=all_cols)
    
    
    for chunk in tau_chunks:
        temp_df=chunk.loc[(chunk['initial_Fe_H']>=feh_min) & (chunk['initial_Fe_H']<=feh_max) &
                          (chunk['initial_mass']>=mass_min) & (chunk['initial_mass']<=mass_max) &
                          (chunk['EEP']>=eep_min) & (chunk['EEP']<=eep_max) &
                          (chunk['S_eff']>=Seff_min) & (chunk['S_eff']<=Seff_max)]
        if len(temp_df)>0:
            if len(tau_df)==0:
                tau_df=temp_df
            else:
                tau_df=pd.concat([tau_df,temp_df])
        
    tau_df=tau_df.set_index(index_cols)

    tau_interp= DFInterpolator(tau_df)
    return tau_interp
