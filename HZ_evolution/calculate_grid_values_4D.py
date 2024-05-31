#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:20:36 2023

@author: ntuchow
"""
#may require a decent amount of ram to run
from isochrones.mist import MISTEvolutionTrackGrid
from hz_utils import ROOT_DIR, OUTPUT_DIR, HZ_evolution_MIST
import numpy as np
import pandas as pd
import time

#import MIST evolutionary track grid and get multi-index
track_grid = MISTEvolutionTrackGrid()
df=track_grid.df

query_str='(initial_mass<=2.0) and (EEP < 605) and (EEP > 5)'

df= df.query(query_str)

index_names= df.index.names
fehs= df.index.levels[0].values
fehs= fehs[4:]
masses= df.index.levels[1].values
mass_arr=masses[:80]
eeps= np.array(range(6,605))
Seff_arr= np.linspace(0.1, 2.1,100)

new_ind= pd.MultiIndex.from_product([fehs,mass_arr,eeps,Seff_arr],names=('initial_Fe_H','initial_mass','EEP','S_eff'))

#%%
tau_df= pd.DataFrame(index=new_ind)
tau_df['tau']=np.nan
tau_df['t_int']=np.nan
tau_df['t_ext']=np.nan

t1=time.perf_counter()
for feh in new_ind.levels[0]:
    print("[Fe/H] = %.2f \n" % feh)
    for mass in new_ind.levels[1]:
        print("Mass=%.2f \n" % mass )
        track=df.xs((feh,mass),level=(0,1))
        for eep in new_ind.levels[2]:
            if (feh,mass,eep) not in df.index:
                continue
            try:
                evol= HZ_evolution_MIST(track, eep,HZ_form="K13_optimistic")
                temp_d_arr=np.sqrt(evol.L[-1]/Seff_arr)
                temp_tau_arr=evol.obj_calc_tau(temp_d_arr,mode='default')
                tau_df.loc[(feh,mass,eep),'tau']=temp_tau_arr
                
                temp_t_int=evol.obj_calc_t_interior(temp_d_arr,mode='default')
                tau_df.loc[(feh,mass,eep),'t_int']=temp_t_int
                
                temp_t_ext=evol.obj_calc_t_exterior(temp_d_arr,mode='default')
                tau_df.loc[(feh,mass,eep),'t_ext']=temp_t_ext
                #takes about 0.011s per iteration
                
                
            except:
                evol= HZ_evolution_MIST(track, eep,HZ_form="K13_optimistic")
                temp_d_arr=np.sqrt(evol.L[-1]/Seff_arr)
                temp_tau_arr=evol.obj_calc_tau(temp_d_arr,mode='coarse')
                tau_df.loc[(feh,mass,eep),'tau']=temp_tau_arr
                
                temp_t_int=evol.obj_calc_t_interior(temp_d_arr,mode='coarse')
                tau_df.loc[(feh,mass,eep),'t_int']=temp_t_int
                
                temp_t_ext=evol.obj_calc_t_exterior(temp_d_arr,mode='coarse')
                tau_df.loc[(feh,mass,eep),'t_ext']=temp_t_ext
                #print("Problem at FeH = %.1f , Mass = %.2f, EEP = %d" %(feh,mass,eep))

t2=time.perf_counter()
print("Took ", (t2-t1), " seconds")
#previously took 5054 seconds
tau_df.to_csv(OUTPUT_DIR+"tau_df_K13_optimistic_4D.csv")
