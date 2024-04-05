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

nages=10
ages= np.linspace(8,9.7,nages)#np.log10(1e9)
dage= 0.01

#%% define priors
#pars=[mass,eep,feh]
prior_arr=[st.norm(loc=0.75,scale=0.05),
           st.uniform(loc=200,scale= 300),
           st.norm(loc= -0.35,scale=0.5)]

ndim=3
nwalkers=32 #32
nsamples=20000 #10000


mist_track = MIST_EvolutionTrack()

samples_arr=[] #array to hold samples

#%%
for i in range(nages):
    
    print("log(age)= ",ages[i])
    target_dict={'Teff':(star_Teff,star_dTeff),
                 'logg':(star_logg,star_dlogg*4),
                 'feh':(star_feh,star_dfeh*2),
                 'logL':(star_logL,star_dlogL),
                 'mass':(star_mass,star_dmass*4),
                 'age':(ages[i],dage)}
    prop_names=list(target_dict.keys())  #['Teff', 'logg', 'feh', 'logL','mass','age']


    global posterior_args
    posterior_args=[target_dict,prop_names, mist_track, prior_arr,[]]

    def ln_post_global(pars):
        ln_post =log_posterior(pars, target_dict, prop_names, 
                               mist_track, prior_arr,[])
        return ln_post
    
    p0=sample_prior(nwalkers, ndim, prior_arr,special_ind=[])

    #for some reason multiprocessing requires this
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_post_global) 
    state=sampler.run_mcmc(p0, 5, progress=False)
    samples=sampler.get_chain()
    sampler.reset()


    if __name__ == '__main__':
        with mp.Pool() as mcmcpool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_post_global, 
                                            pool=mcmcpool)
            sampler.run_mcmc(p0, nsamples, progress=True)

        samples=sampler.get_chain()
    
    flat_samples = sampler.get_chain(discard=600,thin=10, flat=True)
    samples_arr.append(flat_samples)

#%% construct tau interpolator

Seff_min=0.1
Seff_max=2.1

for j in range(nages):
    flat_samples=samples_arr[j]
    if j ==0:
        mass_min=flat_samples[:,0].min()
        mass_max=flat_samples[:,0].max()

        eep_min=flat_samples[:,1].min()
        eep_max=flat_samples[:,1].max()

        feh_min=flat_samples[:,2].min()
        feh_max=flat_samples[:,2].max()
        continue
    
    if flat_samples[:,0].min()< mass_min:
        mass_min=flat_samples[:,0].min()
    if mass_max<flat_samples[:,0].max():
        mass_max=flat_samples[:,0].max()
    if eep_min>flat_samples[:,1].min():
        eep_min=flat_samples[:,1].min()
    if eep_max<flat_samples[:,1].max():
        eep_max=flat_samples[:,1].max()
    if feh_min>flat_samples[:,2].min():
        feh_min=flat_samples[:,2].min()
    if feh_max<flat_samples[:,2].max():
        feh_max=flat_samples[:,2].max()


func_4d=construct_interpolator_4D(feh_min=feh_min,feh_max=feh_max,mass_min= mass_min,mass_max= mass_max,
                           eep_min=eep_min,eep_max=eep_max,Seff_min=0.1,Seff_max=2.1)

#%%
mist_cols=['mass','logL','age']

n_points= samples_arr[0].shape[0]

Seff_arr=np.ones((n_points,nages))*np.nan
age_arr=np.ones((n_points,nages))*np.nan
tau_arr=np.ones((n_points,nages))*np.nan
t_int_arr=np.ones((n_points,nages))*np.nan
t_ext_arr=np.ones((n_points,nages))*np.nan

#K2-18 b properties
Period= 267.29 

for k in range(nages):
    print("log(age)= ",ages[k])
    flat_samples=samples_arr[k]
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
    
        Seff_arr[i,k]=Seff
        age_arr[i,k]=temp_output['age']
        tau_arr[i,k]= temp_tau_arr[0]
        t_int_arr[i,k]= temp_tau_arr[1]
        t_ext_arr[i,k]= temp_tau_arr[2]
 
    
#%%

standard_quants= [0.16, 0.5, 0.84]

for q in range(nages):
    if q==0:
        age_quant_arr= [np.nanquantile(age_arr[:,q], standard_quants)]
        tau_quant_arr= [np.nanquantile(tau_arr[:,q], standard_quants)]
        t_int_quant_arr= [np.nanquantile(t_int_arr[:,q], standard_quants)]
        continue
    
    temp_age=[np.nanquantile(age_arr[:,q], standard_quants)]
    age_quant_arr=np.concatenate([age_quant_arr,temp_age])
    
    temp_tau=[np.nanquantile(tau_arr[:,q], standard_quants)]
    tau_quant_arr=np.concatenate([tau_quant_arr,temp_tau])
    
    temp_t_int=[np.nanquantile(t_int_arr[:,q], standard_quants)]
    t_int_quant_arr=np.concatenate([t_int_quant_arr,temp_t_int])
    
#%%  

#a_arr= age_quant_arr[:,1]
logt_arr= np.log10(tau_quant_arr[:,1])

#t_arr -tau_quant_arr[:,0]
#tau_quant_arr[:,2]-t_arr
logt_err=[logt_arr -np.log10(tau_quant_arr[:,0]),
          np.log10(tau_quant_arr[:,2])-logt_arr]
fig, ax = plt.subplots()

logtint_arr= np.log10(t_int_quant_arr[:,1])
logtint_err=[logtint_arr -np.log10(t_int_quant_arr[:,0]),
          np.log10(t_int_quant_arr[:,2])-logtint_arr]

ax.errorbar(ages,logt_arr,yerr=logt_err,ls='none',marker='o', label= 'log(tau)') 
ax.errorbar(ages,logtint_arr, yerr=logtint_err,ls='none',marker='^', label= 'log(t_int)')
ax.set_xlabel("log(age)")
ax.set_ylabel("log(time)")
ax.legend()

#%%
'''   
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
'''