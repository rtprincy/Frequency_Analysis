from astropy.timeseries import LombScargle
import numpy as np
import pandas as pd

import time
from astropy.io import fits, ascii
from astropy.table import Table, vstack

import os
import glob

import tqdm

from theta_cython import theta_f

def freq_grid(times,oversample_factor=10,f0=None,fn=None):
    times=np.sort(times)
    df = 1.0 / (times.max() - times.min())
    if f0 is None:
        f0 = df
    if fn is None:
        fn = 0.5 / np.median(np.diff(times)) 
    return np.arange(f0, fn, df / oversample_factor)

band='BP'
data=pd.read_csv("hot_hsd_Culpan_plots_frac_plx_ruwe.csv")


Nsample=1000
opt_period_col='opt_period_%s'%(band)
meas_col='freq%s_meas'%(band)
err_col='eperiod_%s'%(band)

data[err_col]=[np.nan]*len(data)
data[opt_period_col]=[np.nan]*len(data)

data[meas_col]=[list(np.ones(Nsample)+np.nan)]*len(data)

source_id=data['source_id'].values

lsp_path="../"
theta_path="../"

oversampling_factor=10



for source in tqdm.tqdm(source_id):
    
    df=pd.read_csv('../astero/Gaia_lightcurve_10000/%s.csv'%(source))
    df=df[df['band']==band]
    df=df[~df['rejected_by_variability']]


    if df.shape[0]>=25: 
        mag=df['mag'].dropna().values
        Time=df['time'].dropna().values

        flux=df['flux'].dropna().values
        flux_err=df['flux_error'].dropna().values
        mag_err= (2.5/np.log(10))*(flux_err / flux)

        freq, lsp=np.load(lsp_path+str(source)+'_%s.npy'%(band))
        theta=np.load(theta_path+str(source)+'_%s.npy'%(band))
        
        psi=2*lsp/theta
        
        ################### Uncertainty estimate ##################################################
        
        # This is to optimise the frequency search around the frequency peak by oversampling the frequency grid. For e.g., if the peak is at 2 c/d, we search for the best frequency from 1.999 to 2.001 using a small frequency step.

        idx_peak=np.argmax(psi)
        
        f_step=np.diff(freq)[0]

        peak_freq=freq[idx_peak]
        
        #Here we take 10 steps before and after the frequency peak as a new frequency search range. 
        
        lower_range=max(0,peak_freq - oversampling_factor*f_step)
        upper_range=peak_freq + oversampling_factor*f_step

        fine_grid_freq=freq_grid(Time,oversample_factor=100,f0=lower_range,fn=upper_range)
    
        lsp_fg = LombScargle(t=Time, y=mag, dy=mag_err,nterms=1).power(frequency=fine_grid_freq, method="cython", normalization="psd")
        theta_fg= theta_f(1/fine_grid_freq, mag, mag_err, Time)
    
        psi_fine_grid=2*lsp_fg/theta_fg
    
        best_freq=fine_grid_freq[np.argmax(psi_fine_grid)]
    
            
        best_period=1/best_freq
        data[opt_period_col][data['source_id']==int(source)]=best_period
    
        freq_sample=np.zeros(Nsample)
    
        # Draw samples from a gaussian distribution around the error and add it to the original magnitude
        # The periodogram is calculated around the frequency peak not the full frequency range, i.e., using fine_grid_freq
        
        for i in range(Nsample):
                
                err_sample=np.random.normal(0,mag_err,mag_err.size)
                
                mag_new=mag+err_sample
                
                lsp_ = LombScargle(t=Time, y=mag_new, dy=mag_err,nterms=1).power(frequency=fine_grid_freq, method="cython", normalization="psd")
            
                theta_ = theta_f(1/fine_grid_freq, mag_new, mag_err, Time)
                
                psi_=2*lsp_/theta_
            
                freq_sample[i]=fine_grid_freq[np.argmax(psi_)]
    
    
        # Frequency and period uncertainties estimate
        
        efreq=np.std(freq_sample) 
        eperiod=efreq/(best_freq**2)
    
        #####################################################################
        data[err_col][data['source_id']==int(source)]=eperiod
        idx=data[meas_col][data['source_id']==int(source)].index[0]
        data[meas_col].iloc[idx]=freq_sample
