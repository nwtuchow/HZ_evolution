#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:57:14 2023

@author: ntuchow
"""

from isochrones.mist import MISTEvolutionTrackGrid,MIST_EvolutionTrack
import utils.hz_utils as hz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from isochrones.interp import DFInterpolator

#%%

#takes ['initial_Fe_H','initial_mass','EEP','S_eff']
def construct_interpolator_4D(feh_min=-2.0,feh_max=0.5,mass_min= 0.1,mass_max= 2.0,
                           eep_min=6,eep_max=605,Seff_min=0.1,Seff_max=2.1):
    index_cols=['initial_Fe_H','initial_mass','EEP','S_eff']
    all_cols=['initial_Fe_H', 'initial_mass', 'EEP', 'S_eff', 'tau', 't_int','t_ext']
    tau_chunks=pd.read_csv("outputs/tau_df_K13_optimistic_4D.csv", chunksize=10**4)
    tau_df= pd.DataFrame(columns=all_cols)
    
    
    for chunk in tau_chunks:
        temp_df=chunk.loc[(chunk['initial_Fe_H']>=feh_min) & (chunk['initial_Fe_H']<=feh_max) &
                          (chunk['initial_mass']>=mass_min) & (chunk['initial_mass']<=mass_max) &
                          (chunk['EEP']>=eep_min) & (chunk['EEP']<=eep_max) &
                          (chunk['S_eff']>=Seff_min) & (chunk['S_eff']<=Seff_max)]
        if len(temp_df)>0:
            tau_df=pd.concat([tau_df,temp_df])
        
    tau_df=tau_df.set_index(index_cols)

    tau_interp= DFInterpolator(tau_df)
    return tau_interp
    
#%%

def construct_interpolator_3D(mass_min= 0.1,mass_max= 2.0, eep_min=6,eep_max=605,
                              Seff_min=0.1,Seff_max=2.1):
    index_cols=['initial_mass','EEP','S_eff']
    all_cols=['initial_mass', 'EEP', 'S_eff', 'tau', 't_int','t_ext']
    tau_chunks=pd.read_csv("outputs/tau_df_K13_optimistic.csv", chunksize=10**4)
    tau_df= pd.DataFrame(columns=all_cols)
    for chunk in tau_chunks:
        temp_df=chunk.loc[(chunk['initial_mass']>=mass_min) & (chunk['initial_mass']<=mass_max) &
                          (chunk['EEP']>=eep_min) & (chunk['EEP']<=eep_max) &
                          (chunk['S_eff']>=Seff_min) & (chunk['S_eff']<=Seff_max)]
        
        if len(temp_df)>0:
            tau_df=pd.concat([tau_df,temp_df])
    
    tau_df=tau_df.set_index(index_cols)

    tau_interp= DFInterpolator(tau_df)
    return tau_interp