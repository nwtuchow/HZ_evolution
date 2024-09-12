"""
Utilities to calculate habitable zones and HZ evolution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import optimize
import astropy.constants as cons
import astropy.units as u
from isochrones.mist import MIST_EvolutionTrack
import pandas as pd
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
OUTPUT_DIR= ROOT_DIR+'/outputs/'

#Constants for HZ polynomials
#Seff_sun, a,b,c,d
#From Kopparapu 2013
c_recent_venus= [1.7763,1.4335e-4,3.3954e-9,-7.6364e-12,-1.1950e-15]
c_runaway_greenhouse=[1.0385,1.2456e-4,1.4612e-8,-7.6345e-12,-1.1950e-15]
c_moist_greenhouse= [1.0146,8.1884e-5,1.9394e-9,-4.3618e-12, -6.8260e-16]
c_max_greenhouse= [0.3507, 5.9578e-5,1.6707e-9,-3.0058e-12,-5.1925e-16]
c_early_mars = [0.3207, 5.4471e-5,1.5275e-9,-2.1709e-12,-3.8282e-16]
#From Ramirez 2018
c_leconte= [1.105, 1.1921e-4, 9.5932e-9, -2.6189e-12, 1.3710e-16]
c_CO2_max= [0.3587,5.8087e-5,1.5393e-9,-8.3547e-13,1.0319e-16]


#solar values for Tuchow 2020 biosignature yield metrics
#from fiducial rotating mist model with approx solar values
#using Kopparapu HZ
SOLAR_B_K13 = {"B_lna_CS": 1.99934863,
           "B_lna_CHZ":	1.55621566,
           "B_lna_2Gyr_CS": 0.84925395,
           "B_lna_2Gyr_CHZ": 0.64869067,
           "B_lna_4Gyr_CS": 0.71611003,
           "B_lna_4Gyr_CHZ": 0.64869067,
           'B_lna_fixed_2Gyr_CS':0.84925395}
#using Ramirez 2018 classical HZ
SOLAR_B_R18 = {"B_lna_CS": 2.1455964,
               "B_lna_CHZ":1.70130507,
               "B_lna_2Gyr_CS":0.9100522,
               "B_lna_fixed_2Gyr_CS":0.9100522}

def hz_boundary(L,Teff,coeff):
    '''
    Calculates a boundary of the habitable zone given the Luminosity and effective
    temperature of the host stars, and coefficients for the hz boundary in question
    

    Parameters
    ----------
    L : float
        stellar Luminosity in Lsun
    Teff : float
        stellar effective temp
    coeff : list of float
        coefficient for polynomial fit for hz boundaries

    Returns
    -------
    d : TYPE
        distance of the hz boundary

    '''
    T= Teff - 5780
    Seff= coeff[0] + coeff[1] * T + coeff[2] * T**2 + coeff[3] * T**3 + coeff[4] * T**4
    d= np.sqrt(L/Seff) #AU
    return d

def hz_flux_boundary(Teff,coeff):
    '''
    Same as hz_boundary, but returns effective flux of boundary
    '''
    T= Teff - 5780
    Seff= coeff[0] + coeff[1] * T + coeff[2] * T**2 + coeff[3] * T**3 + coeff[4] * T**4
    return Seff



G_const= cons.G.to((u.AU**3)/u.Msun/(u.day**2)).value


def P_to_d(Period,Mtot):
    '''calculates planet semimajor axis in AU using keplers 3rd law
    Period in days, Mtot in Msun
    '''
    d= pow(G_const*Mtot*pow(Period,2)/(4*pow(np.pi,2)),(1/3))
    return d
    
def P_to_Seff(Period,Mtot,L):
    ''' calculates Seff of planet given its orbital period
    Period in days, Mtot in Msun, L in Lsun
    '''
    d= P_to_d(Period,Mtot)
    Seff = L / (d**2)
    return Seff
    

class HZ_evolution:
    """This class defines an object for HZ evolution. Given information about
    a star's evolutionary track, this object can calculate derived properties,
    such as the continuously habitable zone and duration spent in the HZ.
    
    Parameters:
    -----------
    age: array
        array of stellar age in yrs
    L: array
        array of Luminosity in Lsun
    Teff: array 
        array of stellar effective temperature in Kelvin
    HZ_form: str, optional
        Formulation for the habitable zone. Currently choose between 
        Kopparapu 2013 moist greenhouse and max greenhouse ("K13"), Kopparapu 
        2013 recent Venus and Early Mars ("K13_optimistic"), Ramirez 2018 ("R18"),
        or define your own HZ formulation selection the "custom" option
    custom_inner_HZ_func: function, optional 
        Custom function for inner HZ boundary in terms of distance,
        taking arguments of L and Teff. Only used when HZ_form="custom"
    custom_outer_HZ_func: function, optional
        Custom function for outer HZ boundary, taking 
        arguments of L and Teff. Only used when HZ_form="custom"
    t0: float
        starting time for calculation in years, ie timescale for planet formation or volatile delivery
    """
    def __init__(self,age,L,Teff,HZ_form="K13",custom_inner_HZ_func=None,
                 custom_outer_HZ_func=None, t0=0.0):
        #need to do something if len<=1?
        self.age=age
        self.L = L
        self.Teff=Teff
        
        custom_HZ=False
        if HZ_form=="K13":
            c_inner =c_moist_greenhouse
            c_outer =c_max_greenhouse
            self.Tmin= 2600.0
            self.Tmax= 7200.0
        elif HZ_form=="K13_optimistic":
            c_inner=c_recent_venus
            c_outer=c_early_mars
            self.Tmin= 2600.0
            self.Tmax= 7200.0
        elif HZ_form=="R18":
            c_inner=c_leconte
            c_outer=c_CO2_max
            self.Tmin= 2600.0
            self.Tmax= 10000.0
        elif HZ_form=='custom':
            custom_HZ=True        
        else:
            print("Invalid HZ")
            return
        
        if t0 >= self.age[-1]:
            self.age = self.age - t0
        elif t0 >0.0:
            ind0=np.where(self.age>t0)[0][0]
            if ind0>0:
                self.age=self.age[(ind0-1):]
                self.L = self.L[(ind0-1):]
                self.Teff = self.Teff[(ind0-1):]
                self.age = self.age - t0
            
        
        if not custom_HZ:
            self.r_inner= hz_boundary(self.L,self.Teff, c_inner)
            self.r_outer= hz_boundary(self.L,self.Teff, c_outer)
        else:
            self.r_inner= custom_inner_HZ_func(self.L,self.Teff)
            self.r_outer= custom_outer_HZ_func(self.L,self.Teff)
        #current hz position
        self.current_i= self.r_inner[-1]
        self.current_o= self.r_outer[-1]
        
        self.S_inner=self.L/pow(self.r_inner,2)
        self.S_outer=self.L/pow(self.r_outer,2)
    
    #calculate Tuchow 2020 def of CHZ
    #needs to be fixed for the case of high eep
    #appears to not work for some low Fe/H cases, might just be outside of temp range for HZ
    def get_sustained_CHZ(self,CHZ_start_age=1e7):
        """
        Calculate Tuchow 2020 definition of the sustained CHZ, the region when planets remain
        within the boundaries of the HZ from a specified starting point until the current day

        Parameters
        ----------
        CHZ_start_age : float, optional
            Age at which sustained habitability is considered to start. Could be zero age main sequence,
            time for the onset of habitability,or timescale for delivery of volatiles to planet
            
        """
        if self.age[-1]<CHZ_start_age:
            self.sCHZ_i= -1
            self.sCHZ_o= -1
            return
           
        CHZ_start_ind=np.where(self.age>CHZ_start_age)[0][0]
        
        #make sure it is bounds for the Kopparapu HZ
        if self.Teff[CHZ_start_ind:].max()> self.Tmax or self.Teff[CHZ_start_ind:].min()<self.Tmin:
            self.sCHZ_i= np.nan
            self.sCHZ_o= np.nan
            return
        
        f_i =interpolate.interp1d(self.age,self.r_inner)
        f_o =interpolate.interp1d(self.age,self.r_outer)

        initial_i = f_i(CHZ_start_age).item()
        initial_o = f_o(CHZ_start_age).item()
    
        ro_min=np.nanmin(self.r_outer[CHZ_start_ind:])

        ri_max=np.nanmax(self.r_inner[CHZ_start_ind:])

        if ro_min <initial_i or ri_max > initial_o or ri_max>ro_min:
            self.sCHZ_i= -1
            self.sCHZ_o= -1
        else:
            self.sCHZ_o= min(ro_min,initial_o)
            self.sCHZ_i= max(ri_max,initial_i)
        return
    
    
    def get_fixed_age_CHZ(self,fixed_age=2.0e9,nd=1000):
        """
        Calculate fixed duration CHZ, also called fixed age CHZ. This is the more 
        standard formulation of the CHZ where planets in the CHZ have spent more than
        a fixed duration in the HZ

        Parameters
        ----------
        fixed_age : float, optional
            Duration of time spent in the habitable zone to be considered continuously habitable.
            Units of yrs
        nd : Int, optional
            number of grid points used to find boundaries of the CHZ
        """
        
        d_range,tau= self.obj_calc_tau_arr(nd=nd,mode="default")
        
        if self.age[-1] < fixed_age or len(self.age)<2:
            self.fCHZ_i = -1
            self.fCHZ_o = -1
            return #self.fCHZ_i, self.fCHZ_o
    
        up_arr=np.array([]) #array for boundaries when goes from non CHZ to CHZ
        down_arr=np.array([]) # boundaries for when goes from CHZ to non CHZ
        #need to do this because in strange cases can have a split CHZ
        
        #needs to be in bounds of temp range of HZ
        if self.Teff.max()> self.Tmax or self.Teff.min()<self.Tmin:
            self.fCHZ_i= np.nan
            self.fCHZ_o= np.nan
            return 
    
        cond_arr= (tau >= fixed_age)
        for i in range(1,nd):
            cond_prev=cond_arr[i-1]
            if (cond_arr[i] ==True) and (cond_prev == False):
                up_arr=np.append(up_arr,d_range[i]) 
            
            if (cond_arr[i]==False) and (cond_prev == True):
                down_arr=np.append(down_arr,d_range[i-1])
    
        if len(up_arr) != len(down_arr):
            print("disagreement between lengths of up_arr and down_arr")
            return 0
    
        if len(up_arr)==1:
            self.fCHZ_i= up_arr[0]
            self.fCHZ_o= down_arr[0]
        elif len(up_arr)==0:
            self.fCHZ_i = -1
            self.fCHZ_o = -1
        else:
            self.fCHZ_i= up_arr
            self.fCHZ_o= down_arr
    
        return     
    
    #needs to be tested
    #may cause crashes on premain sequence and postmain sequence
    def obj_calc_B(self,**kwargs):
        """
        calculates Tuchow 2020 biosignature yield metrics

        Parameters
        ----------
        **kwargs : TYPE
            keyword args for the calc_B function


        Returns
        -------
        B : float
            Biosignature yield metric

        """
        L0 = self.L 
        Teff0 =self.Teff
        age0= self.age
        B= calc_B(L0, Teff0,age0,**kwargs)
        return B
    
    def obj_calc_tau_arr(self,nd=500,**kwargs):
        '''
        Calculates duration spent in the HZ, tau, for an array of distances 
        spread across the HZ. Note: sets regions no longer in HZ to zero
        

        Parameters
        ----------
        nd : int, optional
            number of distance points in array
        **kwargs : TYPE
            obj_calc_tau kwargs

        Returns
        -------
        d_range : numpy.array
            range of distances surrounding the HZ
        tau : numpy.array
            habitable duration, time spent in the habitable zone. 
            Units are the same as units for age

        '''
        
        d_range= np.linspace(0.95*self.current_i,1.05*self.current_o,nd)
        tau=self.obj_calc_tau(d_range,**kwargs)
        
        return d_range, tau
    
    def obj_calc_tau(self,dist,only_CHZ=False,mode="default"):
        '''
        Calculates the duration spent in the habitable zone, tau, for a planet at a 
        given distance. Duration is set to zero if planet is currently outside 
        the HZ. Same as obj_calc_tau_arr but for a single distance

        Parameters
        ----------
        dist : float
            distance between planet and star in AU
        only_CHZ : bool, optional
            sets habitable duration to zero outside CHZ
        mode : TYPE, optional
            Specify how tau is calculated. "default" is smoother but may have 
            problems with tracks that move in and out. "coarse" is faster and 
            works better in problematic areas of evolutionary tracks

        Returns
        -------
        obj_tau : TYPE
            DESCRIPTION.

        '''
        
        if mode=="default":
            obj_tau=calc_tau(dist, self.age, self.r_inner,
                              self.r_outer,only_CHZ=only_CHZ)
        elif mode=="coarse":
            obj_tau=calc_tau_coarse(dist, self.age, self.r_inner,self.r_outer,only_CHZ=only_CHZ)
            
        else:
            print("Invalid mode")
            
        return obj_tau
    
    def obj_calc_t_interior_arr(self,nd=500,mode='default'):
        '''calculate time spent interior to the HZ for array of distances
        

        Parameters
        ----------
        nd : int, optional
            length of distance array
        mode : str, optional
            mode for calculation. Either 'default' or 'coarse'

        Returns
        -------
        d_range : np.array
            array of distances around the HZ
        t_int : np.array
            time spent interior to the HZ

        '''
        d_range= np.linspace(0.95*self.current_i,1.05*self.current_o,nd)
        t_int= self.obj_calc_t_interior(d_range,mode=mode)
        return d_range, t_int
    

    def obj_calc_t_interior(self,dist,mode='default'):
        '''Calculate time spent interior to the HZ for a given distance
        

        Parameters
        ----------
        dist : float
            distance from planet to star (AU)
        mode : str, optional
            mode for calculation. Either 'default' or 'coarse'

        Returns
        -------
        t_int : float
            time spent interior to the HZ.

        '''
        if mode=='default':
            t_int= calc_t_interior(dist, self.age, self.r_inner, self.r_outer)
        elif mode=='coarse':
            t_int= calc_t_interior_coarse(dist, self.age, self.r_inner, self.r_outer)
        else:
            print("Invalid mode")
        
        return t_int
    
    def obj_calc_t_exterior_arr(self,nd=500,mode='default'):
        '''
        Calculates time spent exterior to the HZ for an array of distances

        Parameters
        ----------
        nd : int, optional
            number of distance points around HZ
        mode : str, optional
            mode for calculation. Either 'default' or 'coarse'

        Returns
        -------
        d_range : np.array
            array of distances around the HZ
        t_ext : np.array
            array of time spent exterior to the HZ

        '''
        d_range= np.linspace(0.95*self.current_i,1.05*self.current_o,nd)
        t_ext=self.obj_calc_t_exterior(d_range,mode=mode)
        return d_range, t_ext
        
    
    
    def obj_calc_t_exterior(self,dist,mode='default'):
        '''
        Calculates time spent exterior to the HZ for a specified distance

        Parameters
        ----------
        mode : str, optional
            mode for calculation. Either 'default' or 'coarse'

        Returns
        -------
        t_ext : float
            time spent exterior to the HZ

        '''
        if mode=='default':
            t_ext= calc_t_exterior(dist, self.age, self.r_inner, self.r_outer)
        elif mode=='coarse':
            t_ext= calc_t_exterior_coarse(dist, self.age, self.r_inner, self.r_outer)
        else:
            print("Invalid Mode")
        
        return t_ext
    
    def CHZ_dist_fraction(self,form="sustained"):
        '''calculate fraction of HZ occupied by CHZ
        form: formulation of the CHZ, "sustained" for the Tuchow and Wright 2023 formulation of the sustained CHZ,
            'fixed age' for the fixed age CHZ, planets that spend longer that a given duration in the HZ
        '''
        
        if len(self.age)<=1:
            self.f_d=0
            return self.f_d
        
        if form=="fixed" or form=="fixed age":
            if hasattr(self,'fCHZ_i') and hasattr(self,'fCHZ_o'):
                C_i= self.fCHZ_i
                C_o= self.fCHZ_o
            else:
                print("CHZ undefined")
                self.f_d=np.nan
                return self.f_d
        elif form=="sustained":
            if hasattr(self,'sCHZ_i') and hasattr(self,'sCHZ_o'):
                C_i = self.sCHZ_i
                C_o = self.sCHZ_o
            else:
                print("CHZ undefined")
                self.f_d=np.nan
                return self.f_d
        else:
            print("CHZ undefined")
            self.f_d=np.nan
            return self.f_d
        
        if (self.current_o ==self.current_i) or (self.current_o < self.current_i):
            print("Ill defined HZ")
            self.f_d=np.nan
            return self.f_d
        
        if (C_o==-1) or (C_i==-1):
            self.f_d = 0.0
            return self.f_d
        
        if type(C_o) == np.ndarray or type(C_i) == np.ndarray:
            self.f_d = 0.0
            return self.f_d
        
        if(np.isnan(C_o) or np.isnan(C_i)):
            self.f_d = np.nan
            return self.f_d
        
        if C_o<C_i:
            print("Major problem with CHZ calculation")
            self.f_d=np.nan
            return self.f_d
        
        self.f_d = (C_o-C_i) / (self.current_o - self.current_i)
        return self.f_d
    
    def CHZ_planet_fraction(self, beta=-1, form="sustained"):
        '''calculate fraction of HZ planets in CHZ assuming power law distribution of planets
        beta: powerlaw exponent
        form: form: formulation of the CHZ, "sustained" for the Tuchow and Wright 2023 formulation of the sustained CHZ,
            'fixed age' for the fixed age CHZ, planets that spend longer that a given duration in the HZ'''
        if len(self.age)<=1:
            self.f_p=0
            return self.f_p
        
        if form=="fixed" or form=="fixed age":            
            if hasattr(self,'fCHZ_i') and hasattr(self,'fCHZ_o'):
                C_i= self.fCHZ_i
                C_o= self.fCHZ_o
            else:
                print("CHZ undefined")
                self.f_p=np.nan
                return self.f_p
        elif form=="sustained":
            if hasattr(self,'sCHZ_i') and hasattr(self,'sCHZ_o'):
                C_i = self.sCHZ_i
                C_o = self.sCHZ_o
            else:
                print("CHZ undefined")
                self.f_p=np.nan
                return self.f_p
        else:
            print("CHZ undefined")
            self.f_p=np.nan
            return self.f_p
        
        
        if (self.current_o ==self.current_i) or (self.current_o < self.current_i):
            print("Ill defined HZ")
            self.f_p=np.nan
            return self.f_p
        
        if (C_o==-1) or (C_i==-1):
            self.f_p = 0.0
            return self.f_p
        
        if type(C_o) == np.ndarray or type(C_i) == np.ndarray:
            self.f_p = 0.0
            return self.f_p
        
        if(np.isnan(C_o) or np.isnan(C_i)):
            self.f_d = np.nan
            return self.f_d
        
        if C_o < C_i:
            print("Major problem with CHZ calculation")
            self.f_p=np.nan
            return self.f_p
        
        if beta==-1:
            self.f_p = np.log(C_o/C_i)/np.log(self.current_o/self.current_i)
        else:
            exp_num=beta + 1.0
            self.f_p= (pow(C_o,exp_num)-pow(C_i,exp_num))/  \
                (pow(self.current_o,exp_num)-pow(self.current_i,exp_num))
                
        return self.f_p
    
    
    def plot_HZ(self,CHZ_start_age=1e7,include_sCHZ=False, include_start_age=False):
        '''
        plot the HZ as a function of time

        Parameters
        ----------
        CHZ_start_age : float, optional
            DESCRIPTION. Time where continuous habitability is considered to start in yrs
            The default is 1e7.
        include_sCHZ : bool, optional
            DESCRIPTION. Plot region in the sustained CHZ. The default is False.
        include_start_age : bool, optional
            DESCRIPTION. Plot CHZ start age as a dashed vertical line.
            The default is False.

        Returns
        -------
        pyplot figure and axes
        '''
        if len(self.age)<=1:
            return 0
        hz_fig, hz_ax = plt.subplots()
        hz_ax.plot(self.age,self.r_inner,ls='-',color='black')
        hz_ax.plot(self.age,self.r_outer,ls='-',color='black')
        hz_ax.set_xscale("log")
        hz_ax.set_xlabel("time (yr)")
        hz_ax.set_ylabel("distance (AU)")
        #hz_ax.set_xlim([0.95*CHZ_start_age,self.age[-1]])
        if include_start_age:
            hz_ax.axvline(x=CHZ_start_age, ls='--',color='black')
        
        if (hasattr(self,'sCHZ_i') and hasattr(self,'sCHZ_o')) and include_sCHZ:
            if (self.sCHZ_i!=-1) and (self.sCHZ_o!=-1):
                hz_ax.axhline(y=self.sCHZ_i,ls=':',color='green')
                hz_ax.axhline(y=self.sCHZ_o,ls=':',color='green')
        return hz_fig, hz_ax


class HZ_planet(HZ_evolution):
    """Subclass of HZ_evolution describing the history of a specified planet. 
    Given the position of a planet and its star's evolution, this object computes 
    the instellation history and durations spent in the habitable zone,
    as well as the time spent interior to the habitable zone and time spent exterior to it.
    
    Parameters:
    -----------
    age: array
        array of stellar age in yrs
    L: array
        array of Luminosity in Lsun
    Teff: array 
        array of stellar effective temperature in Kelvin
    Dist: float
        planet star separation in AU (optional, can provide planet's period and stellar mass instead)
    Period: float
        planet's orbital period in days (optional can provide distance instead)
    Mstar: float
        mass of host star in M_sun (only used in Keplers 3rd law to get dist when Period is provided)
    HZ_form: str, optional
        Formulation for the habitable zone. Currently choose between 
        Kopparapu 2013 moist greenhouse and max greenhouse ("K13"), Kopparapu 
        2013 recent Venus and Early Mars ("K13_optimistic"), Ramirez 2018 ("R18"),
        or define your own HZ formulation selection the "custom" option
    custom_inner_HZ_func: function, optional 
        Custom function for inner HZ boundary in terms of distance,
        taking arguments of L and Teff. Only used when HZ_form="custom"
    custom_outer_HZ_func: function, optional
        Custom function for outer HZ boundary, taking 
        arguments of L and Teff. Only used when HZ_form="custom"
    """
    def __init__(self, age,L,Teff,Dist=None, Period=None,Mstar=None,
                 HZ_form="K13",
                 custom_inner_HZ_func=None,
                 custom_outer_HZ_func=None, t0 = 0.0):
        super().__init__(age, L, Teff, HZ_form=HZ_form,
                         custom_inner_HZ_func=custom_inner_HZ_func,
                         custom_outer_HZ_func=custom_outer_HZ_func,
                         t0= t0)
        
        if Dist==None:
            if Period==None or Mstar==None:
                print("Error: Planet Distance not defined")
                return
            else:
                self.Dist= P_to_d(Period, Mstar)
        else:
            self.Dist= Dist
            
        self.Seff= self.L/pow(self.Dist,2)
        self.current_age= self.age[-1]
        self.t0=t0
        
    def get_tau(self, **kwargs):
        '''
        Calculate the duration the planet spends in the habitable zone.
        Takes the same keyword arguments as obj_calc_tau in parent class
        '''
        self.tau=self.obj_calc_tau(self.Dist,**kwargs)[0]
        return self.tau
        
    def get_t_int(self, **kwargs):
        '''
        Calculate the duration the planet spent interior to the habitable zone.
        Takes same kwargs as obj_calc_t_interior in parent class.
        '''
        self.t_int=self.obj_calc_t_interior(self.Dist,**kwargs)[0]
        return self.t_int
        
    def get_t_ext(self, **kwargs):
        '''
        Calculate the duration the planet spent exterior to the habitable zone.
        Takes same kwargs as obj_calc_t_exterior in parent class.
        '''
        self.t_ext=self.obj_calc_t_exterior(self.Dist,**kwargs)[0]
        return self.t_ext
    
    # def classify_planet(self, CHZ_start_age=None):
    #     if (hasattr(self,'sCHZ_i') and hasattr(self,'sCHZ_o')):
    #         pass
    #     elif CHZ_start_age==None:
    #         if self.t0>0 :
    #             self.get_sustained_CHZ(CHZ_start_age=self.t0)
    #         else:
    #             self.get_sustained_CHZ() #use default start time
    #     else:
    #         self.get_sustained_CHZ(CHZ_start_age=CHZ_start_age)
        
        
    #     c_str=''
    #     if (self.Dist >= self.sCHZ_i) and (self.Dist<= self.sCHZ_o):
    #         c_str="Continuously Habitable Zone"
    #     elif (self.Dist < self.sCHZ_i) and (self.Dist >= self.current_i):
    #         c_str="Inner Belatedly Habitable Zone"
    #     elif (self.Dist > self.sCHZ_o) and (self.Dist <= self.current_o):
    #         c_str= "Outer Belatedly Habitable Zone"
    #     else:
    #         c_str="Outside of HZ"
        
    #     print(c_str)
    #     return c_str


def HZ_evolution_MIST(track,eep,**kwargs):
    '''shortcut to turn a MIST evolutionary track (slice of dataframe) into an 
    HZ evolution object
    

    Parameters
    ----------
    track : pandas dataframe
        MIST evolutionary track (slice of model grid).
    eep : int
        Equivalent evolutionary phase, used as a proxy for age in model grid
    **kwargs : 
        keyword arguments for initializing HZ_evolution object

    Returns
    -------
    HZ_evolution
        HZ_evolution object for specified track

    '''
    if eep==0 :
        L=[np.nan]
        Teff=[np.nan]
        age=[0]
    else:
        L = (10**track.logL.loc[:eep]).to_numpy()
        Teff=(10**track.logTeff.loc[:eep]).to_numpy()
        age=(track.star_age.loc[:eep]).to_numpy()
    
    return HZ_evolution(age, L, Teff,**kwargs)

def generate_interpolated_evol_track(pars,track_cols=['age','logL','Teff'],n_eep=400,eep_start=1,mist_track=MIST_EvolutionTrack()):
    '''
    Function to obtain a sythetic evolutionary track of a star 
    via interpolation of MIST models between grid points

    Parameters
    ----------
    pars : np.array
        Array of stellar fundamental properties ( mass(M_sun), EEP, [Fe/H] )
    track_cols : list of str, optional
        List of stellar properties to include in the interpolated track. 
        Must be MIST column names.
        The default is ['age','logL','Teff'].
    n_eep : int, optional
        Number of EEP points between the start date and the current evolutionary phase. 
        The default is 400.
    eep_start : int, optional
        Starting evolutionary phase for track. The default is 1.
    mist_track : isochrones.mist.MIST_EvolutionTrack, optional
        Mist evolutionary track object. Optional, but can specify it to reduce initialization time

    Returns
    -------
    trackdf : pandas dataframe
        dataframe for the star's evolutionary track
    '''
    eep_arr=np.linspace(eep_start,pars[1],n_eep)
    pts= np.empty((3,n_eep))

    pts[0,:]=pars[0]*np.ones(n_eep)
    pts[1,:]=eep_arr
    pts[2,:]=pars[2]*np.ones(n_eep)
    
    trackdf=pd.DataFrame()
    trackdf.index=eep_arr
    trackdf[track_cols]= np.nan*np.ones((n_eep,len(track_cols)))
    for q in range(n_eep):
        trackdf.iloc[q]=mist_track.interp_value(list(pts[:,q]), track_cols)
    
    return trackdf


def calc_tau(d, age, r_inner,r_outer, only_CHZ=False):
    '''calculates the habitable duration of a planet at separation 'd' for star of age 'age'
    r_inner and r_outer are arrays of HZ boundaries in time in AU
    age array gives times for each entry in r_inner and r_outer
    tau will be in same units as age
    It is recommended to use the obj_calc_tau function in the HZ_evolution object
        rather than calling this function by itself
    '''
    if not (isinstance(d,np.ndarray) or isinstance(d,list)):
        d= np.array([d])
    
    nd=len(d)
    
    if age[-1] <0:
        return np.zeros(nd)
    
    if only_CHZ==False:
        ri_func=interpolate.interp1d(age,r_inner)
        ro_func=interpolate.interp1d(age, r_outer) 
    
    ri_min=min(r_inner)
    ro_max=max(r_outer)
    ri_max= max(r_inner) #CHZ inner for t1=0
    ro_min= min(r_outer) #CHZ outer for t1=0
    
    t_life=np.zeros(nd)
    
    for k in range(nd):
        if d[k] < r_inner[-1] or d[k] > r_outer[-1]:
            t_life[k]=0.0 #case of if outside of IHZ
        elif d[k] >= ri_max and d[k]<=ro_min:
            t_life[k] = age[-1]-age[0] # if in CHZ, subtract out age at start time
        else:
            if only_CHZ==True:
                t_life[k]=0.0
                continue
            #for BHZ planets
            inner_age=-1
            outer_age=-1
            if d[k]>= ri_min and d[k]<= ri_max:
                locs=np.argwhere(np.diff(np.sign(r_inner-d[k]))).flatten()
                inner_age0= age[locs[-1]]
                opt_func_i= lambda x: ri_func(x)-d[k]
                inner_age=optimize.root(opt_func_i,inner_age0)
                inner_age=inner_age.x[0]
            if d[k]>= ro_min and d[k]<=ro_max:
                locs=np.argwhere(np.diff(np.sign(r_outer-d[k]))).flatten()
                outer_age0 = age[locs[-1]]
                opt_func_o= lambda x: ro_func(x)-d[k]
                outer_age=optimize.root(opt_func_o,outer_age0)
                outer_age=outer_age.x[0]
            
            if inner_age!=-1 or outer_age!=-1:
                t_life[k]= age[-1]-max(inner_age,outer_age)
            else:
                t_life[k] =0.0
            
    t_life[t_life<0.0]=0.0
            
    return t_life


def calc_tau_coarse(d, age, r_inner,r_outer,only_CHZ=False):
    '''
    #faster and more flexible version of calc_tau
    #currently more coarse than calc_tau, doesn't use interpolation, or optimization to save computation time
    '''
    
    if not (isinstance(d,np.ndarray) or isinstance(d,list)):
        d= np.array([d])
    
    nd=len(d)
    
    if age[-1] <0:
        return np.zeros(nd)
    
    ri_min=min(r_inner)
    ro_max=max(r_outer)
    ri_max= max(r_inner) #CHZ inner for t1=0
    ro_min= min(r_outer) #CHZ outer for t1=0
    
    t_life=np.zeros(nd)
    
    for k in range(nd):
        if d[k] < r_inner[-1] or d[k] > r_outer[-1]:
            t_life[k]=0.0 #case of if outside of current day HZ
        elif d[k] >= ri_max and d[k]<=ro_min:
            t_life[k] = age[-1]-age[0] # if in CHZ, subtract out age at start
        else:
            if only_CHZ==True:
                t_life[k]=0.0
                continue
            #for BHZ planets
            inner_age=-1
            outer_age=-1
            if d[k]>= ri_min and d[k]<= ri_max:
                locs=np.argwhere(np.diff(np.sign(r_inner-d[k]))).flatten()
                inner_age= age[locs[-1]]
            if d[k]>= ro_min and d[k]<=ro_max:
                locs=np.argwhere(np.diff(np.sign(r_outer-d[k]))).flatten()
                outer_age = age[locs[-1]]
            
            if inner_age!=-1 or outer_age!=-1:
                t_life[k]= age[-1]-max(inner_age,outer_age)
            else:
                t_life[k] =0.0
            
    t_life[t_life<0.0]=0.0
            
    return t_life

def calc_t_interior(d, age, r_inner,r_outer):
    '''calculate time spent interior to the habitable zone
    use obj_calc_t_interior in HZ_evolution class instead of calling this function by itself

    '''
    if not (isinstance(d,np.ndarray) or isinstance(d,list)):
        d= np.array([d])
    
    nd=len(d)
    
    if age[-1] <0:
        return np.zeros(nd)
    
    ri_min=min(r_inner)
    ri_max= max(r_inner) 
    
    ri_func= interpolate.interp1d(age,r_inner)
    
    t_interior=np.zeros_like(d)
    
    for q in range(nd):
        if d[q] < r_inner[-1] or d[q] > r_outer[-1]: # set to zero outside HZ
            t_interior[q]=0
        elif d[q] > ri_max:
            t_interior[q]=0
        elif d[q]<=ri_max and d[q]>ri_min:
            locs=np.argwhere(np.diff(np.sign(r_inner-d[q]))).flatten()
            inner_age0= age[locs[-1]] #guess for age when crosses inner boundary
            opt_func= lambda x: ri_func(x)-d[q]
            
            inner_age=optimize.root(opt_func,inner_age0)
            t_interior[q]= inner_age.x[0]
        elif d[q]<=ri_min:
            t_interior[q]=0
    
    t_interior[t_interior<0.0]=0.0
    
    return t_interior


def calc_t_interior_coarse(d, age, r_inner,r_outer):
    '''faster but less precise version of calc_t_interior'''
    if not (isinstance(d,np.ndarray) or isinstance(d,list)):
        d= np.array([d])
    
    nd=len(d)
    
    if age[-1] <0:
        return np.zeros(nd)
    
    
    ri_min=min(r_inner)
    ri_max= max(r_inner) 
    
    t_interior=np.zeros_like(d)
    
    for q in range(nd):
        if d[q] < r_inner[-1] or d[q] > r_outer[-1]: # set to zero outside HZ
            t_interior[q]=0
        elif d[q] > ri_max:
            t_interior[q]=0
        elif d[q]<=ri_max and d[q]>ri_min:
            locs=np.argwhere(np.diff(np.sign(r_inner-d[q]))).flatten()
            inner_age= age[locs[-1]]
            t_interior[q]= inner_age
        elif d[q]<=ri_min:
            t_interior[q]=0
    
    t_interior[t_interior<0.0]=0.0
    
    return t_interior

def calc_t_exterior(d, age, r_inner,r_outer):
    '''function to calculate time spent exterior to the HZ
    use obj_calc_t_exterior in the HZ_evolution object instead of calling this function directly'''    
    
    if not (isinstance(d,np.ndarray) or isinstance(d,list)):
        d= np.array([d])
    
    nd=len(d)
    
    if age[-1] <0:
        return np.zeros(nd)
    
    ro_max=max(r_outer) 
    ro_min= min(r_outer)
    
    ro_func= interpolate.interp1d(age,r_outer)
    
    t_ext=np.zeros_like(d)
    
    for q in range(nd):
        if d[q] < r_inner[-1] or d[q] > r_outer[-1]: # set to zero outside HZ
            t_ext[q]=0
        elif d[q] > ro_max:
            t_ext[q]=0
        elif d[q]<=ro_max and d[q]>ro_min:
            locs=np.argwhere(np.diff(np.sign(r_outer-d[q]))).flatten()
            outer_age0= age[locs[-1]] #guess for age when crosses outer boundary
            opt_func= lambda x: ro_func(x)-d[q]
            
            outer_age=optimize.root(opt_func,outer_age0)
            t_ext[q]= outer_age.x[0]
        elif d[q]<=ro_min:
            t_ext[q]=0
    
    t_ext[t_ext<0.0]=0.0
    
    return t_ext

def calc_t_exterior_coarse(d, age, r_inner,r_outer):
    '''faster but less precision version of calc_t_exterior'''
    if not (isinstance(d,np.ndarray) or isinstance(d,list)):
        d= np.array([d])
    
    nd=len(d)
    
    if age[-1] <0:
        return np.zeros(nd)
    
    ro_max=max(r_outer)
    ro_min= min(r_outer)
    
    t_ext=np.zeros_like(d)
    
    
    for q in range(nd):
        if d[q] < r_inner[-1] or d[q] > r_outer[-1]: # set to zero outside HZ
            t_ext[q]=0
        elif d[q] > ro_max:
            t_ext[q]=0
        elif d[q]<=ro_max and d[q]>ro_min:
            locs=np.argwhere(np.diff(np.sign(r_outer-d[q]))).flatten()
            outer_age= age[locs[-1]] #guess for age when crosses outer boundary
            t_ext[q]= outer_age
        elif d[q]<=ro_min:
            t_ext[q]=0
    
    t_ext[t_ext<0.0]=0.0
    
    return t_ext    


#H for instantaneous HZ
#last index of r_i and r_o needs to be current day hz
def H_instant_HZ(d,r_i,r_o):
    if not (isinstance(d,np.ndarray) or isinstance(d,list)):
        d= np.array([d])
    
    if not (isinstance(r_i,np.ndarray) or isinstance(r_i,list)):
        r_i= np.array([r_i])
        
    if not (isinstance(r_o,np.ndarray) or isinstance(r_o,list)):
        r_o= np.array([r_o])
    
    H_arr= np.zeros(len(d))
    
    for q in range(len(d)):
        if d[q]<r_i[-1]:
            H_arr[q]=0.0
        elif d[q]<=r_o[-1]:
            H_arr[q]=1.0
        elif d[q]> r_o[-1]:
            H_arr[q]=0.0
    
    return H_arr

def H_linear_tau(tau, const=1.0):
    H = const * tau
    return H

def H_exp_tau(tau,b=0.1,A=1.0):
    H = A*(1- np.exp(-b*tau))
    return H

#set H to fixed value if tau greater than a given age
#often seen in the 2Gyr CHZ
def H_fixed_tau(tau, fixed_age=2.0):
    nt=len(tau)
    H =np.zeros(nt)
    for t in range(nt):
        if tau[t]>=fixed_age:
            H[t]=1.9
    return H

#still using old function for tau, change?
def calc_B(L, Teff,age,hab_start_age=1e7,H_form=None,Gamma_form=None,cold_starts=True,
           nd=500, const=1.0,b=0.1, fixed_age=2.0,HZ_form='K13',
           Gamma_norm=1.0, A=1.0):
    '''calculate Tuchow and Wright 2020 biosignature metrics
    age in yrs
    not normalized to solar values, currently doing that afterwards
    hab_start_age in yr
    kwargs const, b, fixed_age for specific versions of H
    HZ form options are K13: Kopparapu 2013, and R18 Ramirez 2018
    old versions should work with cold_starts=True'''
    
    if age[-1] <= hab_start_age:
        B =0.0
        return B
    start_ind= np.where(age>hab_start_age)[0][0]
    
    L=L[start_ind:]
    Teff=Teff[start_ind:]
    age=age[start_ind:]
    
    age=age/1e9 #to Gyr
    #t_0 = hab_start_age/1e9
    if HZ_form=='K13':
        if Teff.max() < 7200.0 and Teff.min() > 2600.0:     #needs to be within limits of Kopparapu HZ formulation
            r_i= hz_boundary(L,Teff,c_moist_greenhouse)
            r_o= hz_boundary(L,Teff,c_max_greenhouse)
        else:
            B=np.nan
            return B
    elif HZ_form == 'R18':
        if Teff.max() <10000.0 and Teff.min() > 2600.0:
            r_i= hz_boundary(L,Teff,c_leconte)
            r_o= hz_boundary(L,Teff,c_CO2_max)
        else:
            B=np.nan
            return B
    else:
        print("Unrecognized HZ form")
        return np.nan
    
    
    d_range= np.linspace(0.95*min(r_i),1.05*max(r_o),nd)
    
    if (H_form==None) or (Gamma_form==None):
        print("No H or Gamma defined")
        return np.nan
    
    if (H_form=='instant') or (H_form=='IHZ'):
        H = H_instant_HZ(d_range,r_i,r_o)
    else:
        #set tau depending on whether cold starts are habitable
        if cold_starts:
            tau=calc_tau_coarse(d_range,age,r_i,r_o,only_CHZ=False)
        else:
            tau= calc_tau_coarse(d_range,age,r_i,r_o,only_CHZ=True)
        
        #set H tau dependence    
        if H_form == 'linear':
            H = H_linear_tau(tau,const=const)
        elif (H_form == 'fixed') or (H_form =='fixed_age'):
            H = H_fixed_tau(tau, fixed_age=fixed_age)
        elif (H_form == 'exp') or (H_form =='exponential'):
            H = H_exp_tau(tau,b=b, A=A)
        elif (H_form=='CS') or (H_form=='cold starts'):
            #outdated, used in older scripts with cold_starts=True
            H= calc_tau(d_range,age,r_i,r_o,t_0=hab_start_age/1e9,only_CHZ=False)
        elif (H_form=='CHZ') or (H_form=='continuous'):
            #outdated, used in older scripts with cold_starts=True
            H= calc_tau(d_range,age,r_i,r_o,t_0=hab_start_age/1e9,only_CHZ=True)
        else:
            print("Unrecognized H")
            return np.nan
    
    if (Gamma_form=='a') or (Gamma_form=='uniform in a'):
        Gamma= Gamma_norm * np.ones(nd)
    elif (Gamma_form=='lna') or (Gamma_form=='uniform in lna'):
        Gamma= Gamma_norm * 1.0/d_range
    else:
        print("Unrecognized Gamma")
        return np.nan
    
    B= np.trapz(H*Gamma,x=d_range)
    return B


def calc_B_MIST(current_eep,track,**kwargs):
    '''same as calc_B but for MIST tracks
    track is slice of MIST tracks dataframe for fixed mass and FeH
    current_eep is eep you want to calculate B for'''
    
    ZAMS_eep=202
    RGB_eep=605 #tip of red giant branch
    #eeps= track.index
    if current_eep <=ZAMS_eep:
        B=np.nan
        return B
    elif current_eep > RGB_eep:
        B=np.nan
        return B
    else:
        L = (10**track.logL.loc[ZAMS_eep:current_eep]).to_numpy()
        Teff=(10**track.logTeff.loc[ZAMS_eep:current_eep]).to_numpy()
        age= (track.star_age.loc[ZAMS_eep:current_eep]).to_numpy()
        B= calc_B(L, Teff,age,**kwargs)
        return B

def calc_B_general(current_eep,track,**kwargs):
    '''general version of calc B for other model grids'''
    ZAMS_eep=202
    RGB_eep=605 #tip of red giant branch
    #eeps= track.index
    if current_eep <=ZAMS_eep:
        B=np.nan
        return B
    elif current_eep > RGB_eep:
        B=np.nan
        return B
    else:
        L = track.loc[ZAMS_eep:current_eep, 'lum'].to_numpy()
        Teff=track.loc[ZAMS_eep:current_eep, 'teff'].to_numpy()
        age= (track.loc[ZAMS_eep:current_eep, 'age'].to_numpy())*1e9
        B= calc_B(L, Teff,age,**kwargs)
        return B



