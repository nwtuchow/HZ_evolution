#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:29:06 2024

@author: ntuchow
"""

from isochrones.mist import MISTEvolutionTrackGrid
import utils.hz_utils as hz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tau_interpolation import construct_interpolator_4D, construct_interpolator_3D



#import stellar tracks from isochrones package
track_grid = MISTEvolutionTrackGrid()
track_df=track_grid.df

index_names= track_df.index.names

#arrays for isochrone grid feh and mass points
fehs= track_df.index.levels[0]
masses= track_df.index.levels[1]

#%% generate a habitable zone evolution object for an example MIST track
#this example is close to a solar analog
test_mass= masses[29] #29 1.0 M_sun
test_feh= fehs[-3] #-3 0.0
test_eep= 355 #355 corresponds to age = 4.58e9 yr
track=track_df.xs((test_feh,test_mass),level=(0,1))

#habitable zone evolution object using Kopparapu et al 2013 habitable zone
evol1= hz.HZ_evolution_MIST(track, test_eep,HZ_form="K13_optimistic")
evol1.get_fixed_age_CHZ(fixed_age=2e9)
evol1.get_sustained_CHZ(CHZ_start_age=1e7)


planet1= hz.HZ_planet(evol1.age, evol1.L, evol1.Teff,
                      Dist=1.0,
                      HZ_form="K13_optimistic")

mode_str='default'
nd=500
S_arr=np.linspace(0.1, 2.0,nd)
d_arr=np.sqrt(evol1.L[-1]/S_arr)


tau_arr= evol1.obj_calc_tau(d_arr,mode=mode_str)

t_int_arr=evol1.obj_calc_t_interior(d_arr,mode=mode_str)

t_ext_arr=evol1.obj_calc_t_exterior(d_arr,mode=mode_str)


print("HZ: ", evol1.current_i, " to ", evol1.current_o)
print("Sustained CHZ: ", evol1.sCHZ_i, " to ", evol1.sCHZ_o)
print("2Gyr fixed duration CHZ: ", evol1.fCHZ_i, " to ",evol1.fCHZ_o)



#%%
hz_fig, hz_ax=evol1.plot_HZ()
#%%
fig, ax = plt.subplots()
ax.plot(S_arr,tau_arr)
ax.invert_xaxis()
ax.set_ylabel('tau')

#%%
fig2, ax2 =plt.subplots()
ax2.plot(S_arr, t_int_arr)
ax2.invert_xaxis()
ax2.set_ylabel("t_int")

#%%
fig3, ax3=plt.subplots()
ax3.plot(S_arr,t_ext_arr)
ax3.invert_xaxis()
ax3.set_ylabel("t_ext")