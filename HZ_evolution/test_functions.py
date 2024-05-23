#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:35:31 2023
test functions
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
test_eep= 400 #355 corresponds to age = 4.58e9 yr
track=track_df.xs((test_feh,test_mass),level=(0,1))

#habitable zone evolution object using Kopparapu et al 2013 habitable zone
evol1= hz.HZ_evolution_MIST(track, test_eep,HZ_form="K13")

tau1=evol1.obj_calc_tau_arr(nd=1000,mode='default')
t_int=evol1.obj_calc_t_interior_arr(nd=1000,mode='default')
evol1.get_fixed_age_CHZ()
evol1.get_sustained_CHZ()


L = (10**track.logL.loc[:test_eep]).to_numpy()
Teff=(10**track.logTeff.loc[:test_eep]).to_numpy()
age=(track.star_age.loc[:test_eep]).to_numpy()

#evol2= hz.HZ_evolution(age, L, Teff,HZ_form="K13_optimistic")
#evol2.obj_calc_tau_arr(output_arr=False)
print("HZ: ", evol1.current_i, " to ", evol1.current_o)
print("Sustained CHZ: ", evol1.sCHZ_i, " to ", evol1.sCHZ_o)
print("2Gyr fixed duration CHZ: ", evol1.fCHZ_i, " to ",evol1.fCHZ_o)
#%%
hz_fig, hz_ax=evol1.plot_HZ()

d_range,tau_arr=evol1.obj_calc_tau_arr()
tau_fig, tau_ax = evol1.plot_tau(d_range,tau_arr)

fig,ax = plt.subplots()
d_range2,t_int_arr=evol1.obj_calc_t_interior_arr()
ax.plot(d_range2,t_int_arr)
ax.set_xlabel("dist")
ax.set_ylabel('t_interior')

#%%
Seff_arr= evol1.L[-1]/ pow(evol1.d_range,2)
S_inner = evol1.L / pow(evol1.r_inner,2)
S_outer = evol1.L / pow(evol1.r_outer,2)

new_S_arr= np.linspace(0.1, 2.1,50)
fine_S_arr=np.linspace(0.1, 2.1,50)
new_d_arr = np.sqrt(evol1.L[-1]/new_S_arr)
new_tau_arr=evol1.obj_calc_tau(new_d_arr,mode='coarse') # timed at 238 us
new_t_int = evol1.obj_calc_t_interior(new_d_arr,mode='coarse') #timed at 212 us

fine_d_arr = np.sqrt(evol1.L[-1]/fine_S_arr) 
fine_tau_arr=evol1.obj_calc_tau(fine_d_arr,mode='default') 
fine_t_int = evol1.obj_calc_t_interior(fine_d_arr,mode='default')

fine_t_ext=evol1.obj_calc_t_exterior(fine_d_arr,mode='default')


#%%
#plt.plot(age,S_inner)
#plt.plot(age,S_outer)

fig2, ax2= plt.subplots()

ax2.plot(new_S_arr,new_tau_arr)
ax2.plot(fine_S_arr,fine_tau_arr)
ax2.invert_xaxis()
ax2.set_ylabel('tau')

fig3, ax3 = plt.subplots()

ax3.plot(new_S_arr,new_t_int)
ax3.plot(fine_S_arr,fine_t_int)
ax3.invert_xaxis()
ax3.set_ylabel('t_int')

fig4, ax4 = plt.subplots()
ax4.plot(fine_S_arr,fine_t_ext)
ax4.invert_xaxis()
ax4.set_ylabel('t_ext')

#%%test interpolators
test_feh=0.0
test_mass=1.11
test_eep=360.2
test_Seff= new_S_arr

feh_min=-0.4
feh_max=0.4

mass_min= 0.8
mass_max= 1.3

eep_min=300
eep_max=400

Seff_min=0.1
Seff_max=2.1

func_3d=construct_interpolator_3D(mass_min= mass_min,mass_max= mass_max, eep_min=eep_min,
                          eep_max=eep_max, Seff_min=0.1,Seff_max=2.1)

func_4d=construct_interpolator_4D(feh_min=feh_min,feh_max=feh_max,mass_min= mass_min,mass_max= mass_max,
                           eep_min=eep_min,eep_max=eep_max,Seff_min=0.1,Seff_max=2.1)

input_arr1=[test_mass*np.ones_like(test_Seff),test_eep*np.ones_like(test_Seff),test_Seff]

input_arr2=[test_feh*np.ones_like(test_Seff),test_mass*np.ones_like(test_Seff),test_eep*np.ones_like(test_Seff),test_Seff]

test_tau_arr1=func_3d(input_arr1,cols=['tau'])
test_tau_arr2= func_4d(input_arr2,cols=['tau'])

#%%
tau_fig,tau_ax= plt.subplots()

tau_ax.plot(test_Seff,test_tau_arr1,label='3D')
tau_ax.plot(test_Seff,test_tau_arr2,label='4D')
tau_ax.set_xlabel('Seff')
tau_ax.set_ylabel('tau')
tau_ax.legend()