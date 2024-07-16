"""
script to calculate habitable durations for each point in the MIST grid
this version is restricted to solar metallicity
use calculate_grid_values_4D.py to add metallicity as an additional dimension
"""

from isochrones.mist import MISTEvolutionTrackGrid
from .hz_utils import ROOT_DIR, OUTPUT_DIR, HZ_evolution_MIST
import numpy as np
import pandas as pd
import time

def calculate_grid_3D(fname="tau_df_K13_optimistic.csv" ,max_mass=2.0,max_EEP=605,
                      min_EEP=5,fixed_feh=0.0,HZ_form="K13_optimistic",verbose=True):
    '''
    calculates habitable durations for the grid of MIST models at specified metallicity
    run this function before calling construct_interpolator_3D
    time consuming and memory intensive, benmarked at 824s

    Parameters
    ----------
    fname : str, optional
        file name for output csv file. The default is "tau_df_K13_optimistic.csv".
    max_mass : float, optional
        max mass for model grid. The default is 2.0.
    max_EEP : int, optional
        Max EEP for model grid. The default is 605.
    min_EEP : int, optional
        Min EEP for model grid. The default is 5.
    fixed_feh : float, optional
        Fixed [Fe/H] value to compute model grid for. Must be a feh value computed by the MIST grid. 
        The default is 0.0.
    HZ_form : str, optional
        Formulation of the HZ used by the HZ_evolution object. The default is "K13_optimistic".
    verbose : bool, optional
        print progress. The default is True.

    Returns
    -------
    None.

    '''
    
    #import MIST evolutionary track grid and get multi-index
    track_grid = MISTEvolutionTrackGrid()
    df=track_grid.df

    index_names= df.index.names
    req_names=['initial_feh','initial_mass','EEP']
    if index_names!=req_names:
        df.index.names=req_names
        
        
    if verbose:
        print("Beginning construction of 3D habitable duration grid. [Fe/H] fixed at %.3f" % fixed_feh)
    
    query_str='(initial_mass<=%f) and (EEP < %d) and (EEP > %d) and (initial_feh==%f)' % (max_mass, max_EEP, min_EEP,fixed_feh)

    df= df.query(query_str)

    fehs= df.index.levels[0].values
    
    if not (fixed_feh in fehs):
        print("Specified [Fe/H] value is not computed as a grid point in the isochrones grid.\nExiting...")
        return
    
    masses= df.index.levels[1].values
    mass_arr=masses[:80]
    eeps= np.array(range(6,605))
    Seff_arr= np.linspace(0.1, 2.1,200)

    new_ind= pd.MultiIndex.from_product([mass_arr,eeps,Seff_arr],names=('initial_mass','EEP','S_eff'))
    
    tau_df= pd.DataFrame(index=new_ind)
    tau_df['tau']=np.nan
    tau_df['t_int']=np.nan
    tau_df['t_ext']=np.nan
    
    t1=time.perf_counter()
    for mass in new_ind.levels[0]:
        if verbose:
            print("Mass=%.2f \n" % mass )
        track=df.xs((fixed_feh,mass),level=(0,1))
        for eep in new_ind.levels[1]:
            if (fixed_feh,mass,eep) not in df.index:
                continue
            try:
                evol= HZ_evolution_MIST(track, eep,HZ_form=HZ_form)
                temp_d_arr=np.sqrt(evol.L[-1]/Seff_arr)
                temp_tau_arr=evol.obj_calc_tau(temp_d_arr,mode='default')
                tau_df.loc[(mass,eep),'tau']=temp_tau_arr
                
                temp_t_int=evol.obj_calc_t_interior(temp_d_arr,mode='default')
                tau_df.loc[(mass,eep),'t_int']=temp_t_int
                
                temp_t_ext=evol.obj_calc_t_exterior(temp_d_arr,mode='default')
                tau_df.loc[(mass,eep),'t_ext']=temp_t_ext
                #takes about 0.011s per iteration
                
                
            except:
                evol= HZ_evolution_MIST(track, eep,HZ_form=HZ_form)
                temp_d_arr=np.sqrt(evol.L[-1]/Seff_arr)
                temp_tau_arr=evol.obj_calc_tau(temp_d_arr,mode='coarse')
                tau_df.loc[(mass,eep),'tau']=temp_tau_arr
            
                temp_t_int=evol.obj_calc_t_interior(temp_d_arr,mode='coarse')
                tau_df.loc[(mass,eep),'t_int']=temp_t_int
            
                temp_t_ext=evol.obj_calc_t_exterior(temp_d_arr,mode='coarse')
                tau_df.loc[(mass,eep),'t_ext']=temp_t_ext
                #print("Problem at FeH = %.1f , Mass = %.2f, EEP = %d" %(feh,mass,eep))

    t2=time.perf_counter()
    if verbose:
        print("Took ", (t2-t1), " seconds")
        print("Saved to: ", OUTPUT_DIR+fname)
    #previously took 827 s
    tau_df.to_csv(OUTPUT_DIR+fname)
    
    return
