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

feh_min=-0.4
feh_max=0.4

mass_min= 0.8
mass_max= 1.2

eep_min=300
eep_max=400

Seff_min=0.1
Seff_max=2.1

index_cols=['initial_Fe_H','initial_mass','EEP','S_eff']
all_cols=['initial_Fe_H', 'initial_mass', 'EEP', 'S_eff', 'tau', 't_int']

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

#%%
tau_interp= DFInterpolator(tau_df)

test_pts= np.array([[0.02,1.05,335,1.0],[0.0,1.0,335.5,1.0]]).T

tau0=tau_interp(test_pts,['tau'])

missingrow_df= tau_df.drop(index=1.0,level=1) #without the solar mass rows
#missingrow_df=tau_df.drop(index=335,level=2) #without a given eep

missingrow_df.index=missingrow_df.index.remove_unused_levels()
missing_interp=DFInterpolator(missingrow_df)

#why is missing row giving nans
#because 1.0 is still in index
#how can I reindex to not have 1.0 be a value
#%%
def test_tau(feh,mass,age,Seff):
    logage=np.log10(age)
    eep=track.get_eep(mass,logage,feh,accurate=True)
    tau_out= tau_interp([feh,mass,eep,Seff])
    return tau_out
    
#%%
#S_arr= np.linspace(0.2,2,50)
S_arr=np.linspace(0.1, 2.1,1000)

test_mass=1.0
test_eep=335
test_feh= 0.0

input_arr= [test_feh*np.ones_like(S_arr),test_mass*np.ones_like(S_arr),test_eep*np.ones_like(S_arr),S_arr]



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
