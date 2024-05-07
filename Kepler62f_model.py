#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:48:07 2023

@author: ntuchow
"""



import sys
import os
os.environ["OMP_NUM_THREADS"] = "1" #prevent numpy from using multiple cores
#os.environ["MKL_NUM_THREADS"] = "1" 
#os.environ["NUMEXPR_NUM_THREADS"] = "1" 
import pandas as pd
import emcee
import numpy as np
from utils.stat_model import log_posterior, sample_prior,model_output
from isochrones.mist import MIST_EvolutionTrack
from isochrones.interp import DFInterpolator
import scipy.stats as st
#from multiprocessing import Pool, cpu_count
import utils.hz_utils as hz
from tau_interpolation import construct_interpolator_4D, construct_interpolator_3D
import matplotlib.pyplot as plt
import isochrones.priors as priors
import multiprocessing as mp
mp.set_start_method('fork')
#%%


#Kepler 62 f properties
star_Teff=5062 #borucki+ 2018
star_dTeff=71
star_logg=4.683 #borucki+ 2018
star_dlogg=0.023
star_feh= -0.34 #fulton and petigura 2018
star_dfeh=0.04
star_logL= -0.5909 #borucki+ 2018
star_dlogL= 0.0077
star_mass=0.764
star_dmass=0.011
target_dict={'Teff':(star_Teff,star_dTeff),
             'logg':(star_logg,star_dlogg*4),
             'feh':(star_feh,star_dfeh*2),
             'logL':(star_logL,star_dlogL),
             'mass':(star_mass,star_dmass*4)}

#%% define priors
#pars=[mass,eep,feh]
prior_arr=[st.norm(loc=0.75,scale=0.05),
           st.uniform(loc=200,scale= 300),  #max eep 350 corresponds to 35 Gyr age for mass upper bound
           st.norm(loc= -0.35,scale=0.5)]

mist_track = MIST_EvolutionTrack()

#%%
test_par= [0.44,250,0.0]
prop_names=['Teff', 'logg', 'feh', 'logL','mass']


global posterior_args
posterior_args=[target_dict,prop_names, mist_track, prior_arr,[]]

def ln_post_global(pars):
    ln_post =log_posterior(pars, target_dict, prop_names, 
                           mist_track, prior_arr,[])
    return ln_post


#%% set dims
ndim=3
nwalkers=32 #32
nsamples=20000 #10000

p0=sample_prior(nwalkers, ndim, prior_arr,special_ind=[])

#%% serial version
sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_post_global) #for some reason multiprocessing requires this
state=sampler.run_mcmc(p0, 5, progress=False)
samples=sampler.get_chain()
sampler.reset()

#%% multiprocessing version (not working for some reason)
if __name__ == '__main__':
    with mp.Pool() as mcmcpool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_post_global, 
                                        pool=mcmcpool)
        sampler.run_mcmc(p0, nsamples, progress=True)

    samples=sampler.get_chain()
    #sampler.reset()
print("done")

#%%
labels = ["mass", "eep", "feh"]


fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()

for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])

axes[-1].set_xlabel("step number")

#tau=sampler.get_autocorr_time()
flat_samples = sampler.get_chain(discard=600,thin=10, flat=True)
#tau=sampler.get_autocorr_time()
import corner
corner_fig = corner.corner(flat_samples,labels=labels,quantiles=[0.16, 0.5, 0.84],show_titles=True)

#%%
np.savetxt('outputs/K62f_sys_mcmc_chain.txt', flat_samples)



#%%
flat_samples=np.loadtxt('outputs/K62f_sys_mcmc_chain.txt')

#%% construct tau interpolator
mass_min=flat_samples[:,0].min()
mass_max=flat_samples[:,0].max()
mass_quants=np.quantile(flat_samples[:,0],[0.16, 0.5, 0.84] )

eep_min=flat_samples[:,1].min()
eep_max=flat_samples[:,1].max()
eep_quants= np.quantile(flat_samples[:,1], [0.16, 0.5, 0.84])

feh_min=flat_samples[:,2].min()
feh_max=flat_samples[:,2].max()
feh_quants= np.quantile(flat_samples[:,2], [0.16, 0.5, 0.84])

Seff_min=0.1
Seff_max=2.1

func_4d=construct_interpolator_4D(feh_min=feh_min,feh_max=feh_max,mass_min= mass_min,mass_max= mass_max,
                           eep_min=eep_min,eep_max=eep_max,Seff_min=0.1,Seff_max=2.1)
#%%
mist_cols=['mass','logL','age']

n_points= flat_samples.shape[0]

Seff_arr=np.ones(n_points)*np.nan
age_arr=np.ones(n_points)*np.nan
tau_arr=np.ones(n_points)*np.nan
t_int_arr=np.ones(n_points)*np.nan
t_ext_arr=np.ones(n_points)*np.nan

#K2-18 b properties
Period= 267.29 

for i in range(n_points):
    #if i%1000==0:
    #    print(i)
    pars=flat_samples[i,:]
    temp_output= model_output(pars,prop_names=mist_cols,mist_track=mist_track)
    L=10**temp_output['logL']
    Seff= hz.P_to_Seff(Period,temp_output['mass'],L)
    #['initial_Fe_H','initial_mass','EEP','S_eff']
    alt_pars=[flat_samples[i,2],flat_samples[i,0],flat_samples[i,1],Seff]
    temp_tau_arr= func_4d(alt_pars)
    
    Seff_arr[i]=Seff
    age_arr[i]=temp_output['age']
    tau_arr[i]= temp_tau_arr[0]
    t_int_arr[i]= temp_tau_arr[1]
    t_ext_arr[i]= temp_tau_arr[2]
    
    
#%%
age_fig, age_ax= plt.subplots()
age_ax.hist(age_arr, bins=40)
age_ax.set_xlabel("log10(age)")

age_quant= np.quantile(10**age_arr,[0.16, 0.5, 0.84] )

#%%
tau_fig, tau_ax= plt.subplots()
tau_ax.hist(np.log10(tau_arr), bins = 40)
tau_ax.set_xlabel('log10(tau)')
tau_quant=np.nanquantile(tau_arr,[0.16, 0.5, 0.84] )

#%%
t_int_fig, t_int_ax =plt.subplots()
t_int_ax.hist(t_int_arr, bins=40)
t_int_ax.set_xlabel('t_interior')
t_int_quant=np.nanquantile(t_int_arr,[0.16, 0.5, 0.84])

#%%
t_ext_fig, t_ext_ax =plt.subplots()
t_ext_ax.hist(t_ext_arr, bins=40)
t_ext_ax.set_xlabel('t_exterior')
t_ext_quant=np.nanquantile(t_ext_arr,[0.16, 0.5, 0.84])


#%%
best_pars= [mass_quants[1],eep_quants[1],feh_quants[1]]
track_cols=['age','logL','Teff']
n_eep=800
'''eep_arr=np.linspace(1,best_pars[1],n_eep)

pts= np.empty((3,n_eep))

pts[0,:]=best_pars[0]*np.ones(n_eep)
pts[1,:]=eep_arr
pts[2,:]=best_pars[2]*np.ones(n_eep)

logL_arr=np.empty(n_eep)
Teff_arr=np.empty(n_eep)
logage_arr= np.empty(n_eep)
for q in range(n_eep):
    temp_output=mist_track.interp_value(list(pts[:,q]), track_cols)
    logage_arr[q]=temp_output[0]
    logL_arr[q]=temp_output[1]
    Teff_arr[q]=temp_output[2]
'''
trackdf=hz.generate_interpolated_evol_track(best_pars,track_cols=track_cols,n_eep=n_eep,mist_track=mist_track)
best_d_planet = hz.P_to_d(Period, best_pars[0]) 

age_input= 10**trackdf['age'].values
L_input= 10**trackdf['logL'].values
Teff_input= trackdf['Teff'].values
best_planet_obj= hz.HZ_planet(age_input,L_input,Teff_input,Dist=best_d_planet,
                         HZ_form="K13_optimistic")  
best_time_bp= age_input[-1] -age_input

best_S_arr=best_planet_obj.Seff


#%% make loop to get Seff
ntracks=25
time_arr= np.empty((ntracks,n_eep))
S_eff_arr= np.empty((ntracks,n_eep))
    
rand_inds=np.random.randint(len(flat_samples),size=ntracks)

for q in range(ntracks):
    ind= rand_inds[q]
    pars=flat_samples[ind,:]
    d_planet = hz.P_to_d(Period, pars[0])
    temptrack=hz.generate_interpolated_evol_track(pars,track_cols=track_cols,n_eep=n_eep,mist_track=mist_track)
    age_input= 10**temptrack['age'].values
    L_input= 10**temptrack['logL'].values
    Teff_input= temptrack['Teff'].values
    temp_planet=hz.HZ_planet(age_input,L_input,Teff_input,Dist=d_planet,
                             HZ_form="K13_optimistic")
    time_bp= age_input[-1] -age_input
    time_arr[q,:]= time_bp

    S_arr=best_planet_obj.Seff
    S_eff_arr[q,:]= S_arr
    
#could plug into Planet object

#temp_output= model_output(pars,prop_names=mist_cols,mist_track=mist_track)
#L=10**temp_output['logL']
#eff= hz.P_to_Seff(Period,temp_output['mass'],L)

hz_inner_flux= hz.hz_flux_boundary(best_planet_obj.Teff,hz.c_recent_venus)
hz_outer_flux= hz.hz_flux_boundary(best_planet_obj.Teff,hz.c_early_mars)

#%%plotting tracks

S_fig, S_ax = plt.subplots()


for j in range(ntracks):
    S_ax.plot(time_arr[j,:],S_eff_arr[j,:],color='gray',alpha=0.5)

S_ax.plot(best_time_bp,best_S_arr,color='black',lw=2)

S_ax.plot(best_time_bp,hz_inner_flux,color='green',ls='--')
S_ax.plot(best_time_bp,hz_outer_flux,color='green',ls='--')

S_ax.invert_xaxis()
S_ax.set_xlabel("Time before present (yr)")
S_ax.set_ylabel("S_eff")
S_ax.set_ylim([0.0,2.1])
S_ax.set_xscale('log')


hz_fig, hz_ax= best_planet_obj.plot_HZ()
hz_ax.axhline(y=best_d_planet,ls='--')
hz_ax.set_ylim([0,2])
hz_ax.set_xlim([1e5,7e8])