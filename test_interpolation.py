#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:15:06 2023

@author: ntuchow
"""

from isochrones.mist import MISTEvolutionTrackGrid,MIST_EvolutionTrack
import utils.hz_utils as hz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from isochrones.interp import DFInterpolator

track_grid = MISTEvolutionTrackGrid()
track= MIST_EvolutionTrack()
# use .get_eep(mass,log(age),feh,accurate=True)

tau_df= pd.read_csv("outputs/tau_df_K13_optimistic.csv", index_col=['initial_mass','EEP','S_eff'])

#%%
tau_interp= DFInterpolator(tau_df)

test_pts= np.array([[0.4,335,1.0],[1.0,335,1.0]]).T

tau0=tau_interp(test_pts,['tau'])

missingrow_df= tau_df.drop(index=0.4,level=0) #without the solar mass rows
#missingrow_df=tau_df.drop(index=335,level=1) #without a given eep

missingrow_df.index=missingrow_df.index.remove_unused_levels()
missing_interp=DFInterpolator(missingrow_df)

#why is missing row giving nans
#because 1.0 is still in index
#how can I reindex to not have 1.0 be a value
#%%
def test_tau(mass,age,Seff):
    feh=0
    logage=np.log10(age)
    eep=track.get_eep(mass,logage,feh,accurate=True)
    tau_out= tau_interp([mass,eep,Seff])
    return tau_out
    
#%%
#S_arr= np.linspace(0.2,2,50)
S_arr=np.linspace(0.1, 2.1,1000)

test_mass=0.4
test_eep=335
test_feh=0

input_arr= [test_mass*np.ones_like(S_arr),test_eep*np.ones_like(S_arr),S_arr]



output_arr= tau_interp(input_arr,['tau','t_int'])
mist_output=track_grid.interp([test_feh,test_mass,test_eep],['age','Teff','logL'])
age=mist_output[0]
Teff=mist_output[1]
Lum= 10**mist_output[2]

output_arr2=missing_interp(input_arr,['tau','t_int'])

S_inner=hz.hz_flux_boundary(Lum, Teff, hz.c_recent_venus)
S_outer= hz.hz_flux_boundary(Lum, Teff, hz.c_early_mars)

d_arr= np.sqrt(Lum/S_arr)
'''
fig, ax =plt.subplots()
ax.plot(d_arr,output_arr[:,0])

fig2, ax2 = plt.subplots()
ax2.plot(d_arr,output_arr[:,1])'''

#%%
fig, ax= plt.subplots()
ax.plot(S_arr,output_arr[:,0])
ax.plot(S_arr,output_arr2[:,0],label='omitting row')
ax.axvline(x=S_inner,ls='--',color='black')
ax.axvline(x=S_outer,ls='--',color='black')

ax.set_xlabel("S_eff")
ax.set_ylabel("tau")
ax.legend()

fig2, ax2 = plt.subplots()
ax2.plot(S_arr,output_arr[:,1])
ax2.plot(S_arr,output_arr2[:,1],label='omitting row')
ax2.set_xlabel("S_eff")
ax2.set_ylabel("t_int")
ax2.axvline(x=S_inner,ls='--',color='black')
ax2.axvline(x=S_outer,ls='--',color='black')
ax2.legend()
