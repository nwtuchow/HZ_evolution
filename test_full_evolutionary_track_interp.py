#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:43:01 2024

@author: ntuchow
"""

from isochrones.mist import MISTEvolutionTrackGrid, MIST_EvolutionTrack
import utils.hz_utils as hz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from isochrones.interp import DFInterpolator

track_grid = MISTEvolutionTrackGrid()

mist_df=track_grid.df

track1= mist_df.xs((0,0.98),level=(0,1))
track2= mist_df.xs((0,1.00),level=(0,1))
track3= mist_df.xs((0,1.02),level=(0,1))

#%% make new dataframe missing row with M=1.0


mist_df2=mist_df.drop(index=1.0,level=1)
mist_df2.index=mist_df2.index.remove_unused_levels()

missing_interp=DFInterpolator(mist_df2)

n_eep=800
eep_arr=np.linspace(50,800,n_eep)#np.arange(n_eep)
pts= [np.zeros_like(eep_arr),1.00*np.ones_like(eep_arr),eep_arr]

interp_track=missing_interp(pts,['age','Teff','logL'])
#%%
fig, ax =plt.subplots()
ax.plot(track1['age'],track1['logL'],label='M=0.98')
ax.plot(track2['age'],track2['logL'],label='M=1.00')
ax.plot(track3['age'],track3['logL'],label='M=1.02')
ax.set_xlabel("log(age)")
ax.set_ylabel("logL")
ax.legend()

#%%
fig2, ax2 =plt.subplots()
ax2.plot(track1['Teff'],track1['logL'],label='M=0.98')
ax2.plot(track2['Teff'],track2['logL'],label='M=1.00')
ax2.plot(track3['Teff'],track3['logL'],label='M=1.02')
ax2.set_xlabel("Teff")
ax2.set_ylabel("logL")
ax2.invert_xaxis()
ax2.legend()

#%%
fig3, ax3 = plt.subplots()
ax3.plot(track2['Teff'],track2['logL'],label='M=1.00')
ax3.plot(interp_track[:,1],interp_track[:,2], label="Interpolated track")

ax3.set_xlabel("Teff")
ax3.set_ylabel("log(L)")
ax3.legend()