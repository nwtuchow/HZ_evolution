#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 17:11:46 2020
statistical model
@author: nxt5109
"""


from isochrones.mist import MIST_EvolutionTrack
import numpy as np
import isochrones.priors as priors
from scipy.stats import uniform

LOG_ONE_OVER_ROOT_2PI = np.log(1.0 / np.sqrt(2 * np.pi))

#note that priors may have different attributes
prior_arr0=[priors.ChabrierPrior(),
           uniform(loc=150,scale=600),
           uniform(loc=-4.0,scale=4.5)]

#log gaussian likelihood
def log_gauss(p,dp,model_p):
    diff=p - model_p
    return LOG_ONE_OVER_ROOT_2PI+np.log(dp) -0.5*diff*diff/(dp*dp)

'''#use scipy version instead
def uniform_dist(x,bounds=(0,1)):
    diff =bounds[1]-bounds[0]
    f=0
    if x>=bounds[0] and x<=bounds[1]:
        f=1./diff
    return f
'''

 #pars=[mass,eep,feh]   
#prop names are properties we age fitting to
def model_output(pars,prop_names=['Teff','logg','feh','logL'],mist_track=None):
    if mist_track==None:
        mist_track=MIST_EvolutionTrack()
        print('Mist track not given as input!')
    
    if type(pars)==np.ndarray:
        pars=pars.tolist()
    mistdict={}
    if len(prop_names)>0:
        temp1=mist_track.interp_value(pars, prop_names) #give same results as true_props
        for n in range(len(prop_names)):
            mistdict[prop_names[n]]=temp1[n]
    
    return mistdict

#compute absolute bolometric luminosity
def L_to_Mbol(lum,MbolSun=4.74):
    Mbol= -2.5*np.log10(lum)+MbolSun
    return Mbol



#pars=[mass,eep,feh]

#takes target_dict of 'key' : (value, uncertainty)
#should add failsafes for nans
def log_likelihood(pars,target_dict,prop_names,mist_track):
    outdict=model_output(pars, prop_names=prop_names,mist_track=mist_track)
    ln_like=0.0
    for prop in target_dict.keys():
        #print(prop)
        if prop in outdict.keys():
            ln_like+=log_gauss(outdict[prop],target_dict[prop][1],
                               target_dict[prop][0])
        
    return ln_like


def log_prior(pars,special_ind,prior_arr=prior_arr0):
    
    lnprior= 0.0
    
    for i in range(len(pars)):
        if i in special_ind: #mass prior has different attributes
            lnprior+=prior_arr[i]._lnpdf(pars[i])
            continue
        lnprior+=prior_arr[i].logpdf(pars[i])
    return lnprior

def log_posterior(pars,target_dict,prop_names,mist_track,prior_arr,special_ind):
    try:
        tot= log_prior(pars,special_ind,prior_arr=prior_arr) + \
            log_likelihood(pars,target_dict,prop_names,mist_track)
    except:
        tot=-np.inf
    else:
        if np.isnan(tot):
            tot= -np.inf
    
    return tot


#draw random samples from prior distribution
#special_ind indicates index of distribution from isochrones package rather than scipy
def sample_prior(nwalkers,ndim,prior_arr,special_ind=[0]):
    p0=np.empty((nwalkers,ndim))
    for i in range(ndim):
        if i in special_ind:
            p0[:,i]=prior_arr[i].sample(nwalkers)
            continue
        p0[:,i]=prior_arr[i].rvs(size=nwalkers)
    
    return p0

#pars=[mass,eep,feh,distance,A_v]
#parallax in mas
#takes target_dict of 'key' : (value, uncertainty)

def fitting_func(pars,target_dict,band_names,prop_names,mist_track,
                 scale):
    if scale==None:
        scale=np.ones(len(target_dict))
        
    outdict=model_output(pars, band_names, prop_names,
                              mist_track=mist_track)
    tot=0.0
    
    props=list(target_dict.keys())
    for i in range(len(props)):
        prop=props[i]
        if prop in outdict.keys():
            tot+= (outdict[prop]-target_dict[prop][0])**2  / scale[i]**2
            
        if prop=='parallax':
            para= 1000./pars[3]
            tot+= (para-target_dict[prop][0])**2  / scale[i]**2
    return tot


