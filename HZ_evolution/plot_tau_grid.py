#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:55:23 2024

@author: ntuchow
"""

from isochrones.mist import MISTEvolutionTrackGrid,MIST_EvolutionTrack
import hz_utils as hz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from isochrones.interp import DFInterpolator

import matplotlib.tri as tri
from scipy.ndimage import gaussian_filter
#from matplotlib import cm, ticker
from scipy.interpolate import interp1d

#%% import tau_df

index_cols=['initial_mass','EEP','S_eff']
all_cols=['initial_mass', 'EEP', 'S_eff', 'tau', 't_int','t_ext']
tau_chunks=pd.read_csv("outputs/tau_df_K13_optimistic.csv", chunksize=10**4)
tau_df= pd.DataFrame(columns=all_cols)

mass_min= 0.1
mass_max= 2.0 
eep_min=6
eep_max=605
Seff_min=0.1
Seff_max=2.1

for chunk in tau_chunks:
    temp_df=chunk.loc[(chunk['initial_mass']>=mass_min) & (chunk['initial_mass']<=mass_max) &
                      (chunk['EEP']>=eep_min) & (chunk['EEP']<=eep_max) &
                      (chunk['S_eff']>=Seff_min) & (chunk['S_eff']<=Seff_max)]
    
    if len(temp_df)>0:
        tau_df=pd.concat([tau_df,temp_df])

tau_df=tau_df.set_index(index_cols)

#%%

track_grid = MISTEvolutionTrackGrid()
track_df=track_grid.df
fehs= track_df.index.levels[0]
masses= track_df.index.levels[1]

#Seff index 90 closest to 1
# 65  is 0.753
#114 is 1.246

S_inds=tau_df.index.levels[2]
Seff_target=S_inds[90]

def Seff_slice(Seff_target):
    grid_df=tau_df.xs(Seff_target,level=2)

    #add feh as dimension to index
    grid_df=pd.concat([grid_df],keys=[0.0],names=['initial_feh'])

    grid_df['star_age']=np.nan
    
    #np.isin function too slow for some reason
    #mask = np.in1d(grid_df.index.values,track_df.index.values)
    for ind in grid_df.index:
        if ind in track_df.index:
            grid_df.loc[ind,'star_age']=track_df.loc[ind,'star_age']
    return grid_df

#%%
grid1= Seff_slice(Seff_target)

age_cutoff= grid1.loc[grid1.index.get_level_values(2)==604,'star_age']
mass_range=age_cutoff.index.levels[1]


f_cutoff= interp1d(mass_range.to_numpy(), age_cutoff.to_numpy(na_value=np.inf))

#%%
mass_arr=grid1.index.get_level_values(1).to_numpy()
age_arr= grid1['star_age'].to_numpy()/1e9 #Gyr
tau_arr=grid1['tau'].to_numpy()
t_int_arr= grid1['t_int'].to_numpy()
t_ext_arr=grid1['t_ext'].to_numpy()


mask=np.logical_not(np.isnan(age_arr))
mass_arr=mass_arr[mask]
age_arr=age_arr[mask]
tau_arr=tau_arr[mask]
t_int_arr=t_int_arr[mask]
t_ext_arr=t_ext_arr[mask]

#tau_arr=np.nan_to_num(tau_arr,nan=0.0)
#t_int_arr=np.nan_to_num(t_int_arr,nan=0.0)
#t_ext_arr=np.nan_to_num(t_ext_arr,nan=0.0)



triang = tri.Triangulation(mass_arr, age_arr)

tau_interp=tri.LinearTriInterpolator(triang,tau_arr)
tint_interp=tri.LinearTriInterpolator(triang,t_int_arr)
text_interp=tri.LinearTriInterpolator(triang,t_ext_arr)





npt=200
M= np.linspace(0.1,2.0,npt)
A=np.linspace(0.0, 10.0,npt)
Mgrid, Agrid = np.meshgrid(M,A)


mask2 = ((Agrid*1e9) > f_cutoff(Mgrid))
#for i in range(npt):
#    for j in range(npt):
#        a_max= f_cutoff(Mgrid[])
#        if A[i,j] > 

#mask2=np.logical_and(Agrid>2,Mgrid>1.70)

TAU= tau_interp(Mgrid,Agrid)
T_INT=tint_interp(Mgrid,Agrid)
T_EXT=text_interp(Mgrid,Agrid)

TAU[mask2]=np.ma.masked
T_INT[mask2]=np.ma.masked
T_EXT[mask2]=np.ma.masked
#TAU= np.ma.filled(TAU,fill_value=0)


#%%


#%%
nlvl=20
#cmapname='Blues' #"Blues_r
#cmapname='Blues_r'
fsize=16

fig, ax = plt.subplots()
cs=ax.contourf(M,A,np.log10(TAU), levels=nlvl)
cbar = fig.colorbar(cs, ax=ax, shrink=0.95)
ax.set_xlabel("Mass (Msun)", fontsize=fsize)
ax.set_ylabel("Age (Gyr)", fontsize=fsize)

#%%

fig2, ax2 = plt.subplots()
cs2=ax2.contourf(M,A,np.log10(T_INT), levels=nlvl)
cbar2 = fig.colorbar(cs2, ax=ax2, shrink=0.95)
ax2.set_xlabel("Mass (Msun)", fontsize=fsize)
ax2.set_ylabel("Age (Gyr)", fontsize=fsize)

#%%

fig3, ax3 = plt.subplots()
cs3=ax3.contourf(M,A,np.log10(T_EXT), levels=nlvl)
cbar3 = fig.colorbar(cs3, ax=ax3, shrink=0.95)
ax3.set_xlabel("Mass (Msun)", fontsize=fsize)
ax3.set_ylabel("Age (Gyr)", fontsize=fsize)
