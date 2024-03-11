#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:20:36 2023

@author: ntuchow
"""

from isochrones.mist import MISTEvolutionTrackGrid
import utils.hz_utils as hz
import numpy as np
import pandas as pd
import time

#import MIST evolutionary track grid and get multi-index
track_grid = MISTEvolutionTrackGrid()
df=track_grid.df



query_str='(initial_mass<=2.0) and (EEP < 605) and (EEP > 5) and (initial_feh==0.0)'

df= df.query(query_str)
#track_df= df.loc[(df['initial_mass']<5.0) & (df['EEP']<605) & (df['EEP']>5)  \
#       & (df['initial_feh']>=-1.0)]

index_names= df.index.names
fehs= df.index.levels[0].values
masses= df.index.levels[1].values
mass_arr=masses[:80]
eeps= np.array(range(6,605))
Seff_arr= np.linspace(0.1, 2.1,200)

new_ind= pd.MultiIndex.from_product([mass_arr,eeps,Seff_arr],names=('initial_mass','EEP','S_eff'))

tau_df= pd.DataFrame(index=new_ind)
tau_df['tau']=np.nan
tau_df['t_int']=np.nan
tau_df['t_ext']=np.nan

feh=0
t1=time.perf_counter()
for mass in new_ind.levels[0]:
    print("Mass=%.2f \n" % mass )
    track=df.xs((feh,mass),level=(0,1))
    for eep in new_ind.levels[1]:
        try:
            evol= hz.HZ_evolution_MIST(track, eep,HZ_form="K13_optimistic")
            temp_d_arr=np.sqrt(evol.L[-1]/Seff_arr)
            temp_tau_arr=evol.obj_calc_tau(temp_d_arr,mode='default')
            tau_df.loc[(mass,eep),'tau']=temp_tau_arr
            
            temp_t_int=evol.obj_calc_t_interior(temp_d_arr,mode='default')
            tau_df.loc[(mass,eep),'t_int']=temp_t_int
            
            temp_t_ext=evol.obj_calc_t_exterior(temp_d_arr,mode='default')
            tau_df.loc[(mass,eep),'t_ext']=temp_t_ext
            #takes about 0.011s per iteration
            
            
        except:
            evol= hz.HZ_evolution_MIST(track, eep,HZ_form="K13_optimistic")
            temp_d_arr=np.sqrt(evol.L[-1]/Seff_arr)
            temp_tau_arr=evol.obj_calc_tau(temp_d_arr,mode='coarse')
            tau_df.loc[(mass,eep),'tau']=temp_tau_arr
            
            temp_t_int=evol.obj_calc_t_interior(temp_d_arr,mode='coarse')
            tau_df.loc[(mass,eep),'t_int']=temp_t_int
            
            temp_t_ext=evol.obj_calc_t_exterior(temp_d_arr,mode='coarse')
            tau_df.loc[(mass,eep),'t_ext']=temp_t_ext
            print("Problem at FeH = %.1f , Mass = %.2f, EEP = %d" %(feh,mass,eep))

t2=time.perf_counter()
print("Took ", (t2-t1), " seconds")
#previously took 827 s
tau_df.to_csv("outputs/tau_df_K13_optimistic.csv")
