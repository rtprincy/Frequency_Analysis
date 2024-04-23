import argparse
import csv

from astropy.io import fits, ascii
from astropy.table import Table, vstack
import argparse
import os
import tqdm

import numpy as np
import pandas as pd
from astropy import time, coordinates as coord, units as u
pd.options.mode.chained_assignment = None
from distutils.util import strtobool
import pwkit.pdm as pdm


def theta_f(periods, mag, magerr, mjd, chunk_size=1000):
    num_periods = len(periods)
    theta_array = np.zeros(num_periods)
    
    for start in range(0, num_periods, chunk_size):
        end = min(start + chunk_size, num_periods)
        
        chunk_periods = periods[start:end]
        chunk_theta = np.zeros_like(chunk_periods)
        chunk_mag = np.tile(mag, (len(chunk_periods), 1))
        chunk_magerr = np.tile(magerr, (len(chunk_periods), 1))
        chunk_mjd = np.tile(mjd, (len(chunk_periods), 1))
        
        phi = chunk_mjd / chunk_periods[:, np.newaxis]
        phi -= np.floor(phi)
        
        sorted_mag = np.zeros_like(chunk_mag)
        sorted_magerr = np.zeros_like(chunk_magerr)
        
        for i in range(len(chunk_periods)):
            idx = np.argsort(phi[i])
            sorted_mag[i] = chunk_mag[i, idx]
            sorted_magerr[i] = chunk_magerr[i, idx]
        
        diff_mag = sorted_mag[:, 1:] - sorted_mag[:, :-1]
        diff_magerr = sorted_magerr[:, 1:]**2 + sorted_magerr[:, :-1]**2
        w = 1 / diff_magerr
        
        numerator = np.sum(w * diff_mag**2, axis=1)
        mean_mag = np.mean(sorted_mag[:, 1:], axis=1, keepdims=True)
        denominator = np.sum((sorted_mag[:, 1:] - mean_mag)**2, axis=1) * np.sum(w, axis=1)
        chunk_theta = numerator / denominator
        
        theta_array[start:end] = chunk_theta
        
    return theta_array


def compute_rms(x_mag):
    return np.sqrt(sum(x_mag**2)/x_mag.size)


def scale_amplitude(x_mag,q_rms):
    #scale any magnitudes to have the same amplitude as qmag
    x_rms=compute_rms(x_mag)
    x_scaled=x_mag*(q_rms/x_rms) 
    x_scaled= x_scaled - np.median(x_scaled)
    return x_scaled

def freq_grid(times,oversampling_factor,f0=None,fn=None):
    times=np.sort(times)
    tbase=(times.max() - times.min())
    df = 4.0 / tbase
    if f0 is None:
        f0 = df
    if fn is None:
        fn = 0.5 / np.mean(np.diff(times)) 
    return np.arange(f0, fn, 1/(tbase * oversampling_factor))

def freq_grid_asas(times,oversampling_factor,f0=None,fn=None):
    times=np.sort(times)
    tbase=(times.max() - times.min())
    df = 4.0 / tbase
    if f0 is None:
        f0 = df
    if fn is None:
        fn = 0.5 / np.mean(np.diff(times)) 
    return np.arange(f0, fn, 1/(tbase * oversampling_factor))


def freq_grid_optimise(n_best_freq,oversampling_factor,psi,lspw,freq):
    
    new_freq=np.zeros(n_best_freq)  
    
    for i in range(n_best_freq):
        idx_best=np.argmax(lspw)
        best_freq=freq[idx_best]
        psi[max(0,idx_best-2*oversampling_factor):idx_best+2*oversampling_factor]=0
        lspw[max(0,idx_best-2*oversampling_factor):idx_best+2*oversampling_factor]=0

    for i in range(n_best_freq):
        idx_best=np.argmax(psi)
        best_freq=freq[idx_best]
        new_freq[i]=best_freq
        psi[max(0,idx_best-2*oversampling_factor):idx_best+2*oversampling_factor]=0

    freq_step=np.diff(freq)[0]
    
    min_freq=new_freq-freq_step
    max_freq=new_freq + freq_step
    
    fine_step=freq_step/oversampling_factor
    
    freq_grid=np.array([])
    
    for i in range(len(new_freq)):
        temp_freq=np.arange(min_freq[i],max_freq[i]+fine_step,fine_step)
        freq_grid=np.hstack([freq_grid,temp_freq])
    
    freq_grid=freq_grid[(freq_grid<0.99)|(freq_grid>1.01)]
    freq_grid=freq_grid[(freq_grid<1.99)|(freq_grid>2.01)]
    freq_grid=freq_grid[(freq_grid<0.49)|(freq_grid>0.51)]
    freq_grid=freq_grid[(freq_grid<0.033)|(freq_grid>0.034)]

    return freq_grid

def remove_outliers(df,filters,mag_col,filter_col):
    dframe=pd.DataFrame()
    for Filter in filters:
        
        temp=df[df[filter_col]==Filter]
        
        # q3, q1 = np.percentile(temp[mag_col], [75 ,25])
        # q99, q01 = np.percentile(temp[mag_col], [99 ,1])
        # iqr=q3-q1
        # h=q3+iqr*1.5
        # l=q1-iqr*1.5
        # temp=temp[(temp[mag_col]<=h)&(temp[mag_col]>=l)]
        # temp=temp[(temp[mag_col]<=q99)&(temp[mag_col]>=q01)]
        
        
        q3_err, q1_err = np.percentile(temp['MAGERR_OPT'], [75 ,25])
        iqr_err=q3_err-q1_err
        h_err=q3_err+iqr_err*1.5
            
        temp=temp[temp['MAGERR_OPT']<=h_err]
                
        dframe=pd.concat([dframe,temp])
        
        
        
    return dframe

def remove_outliers_asas(df,filters,mag_col,filter_col):
    dframe=pd.DataFrame()
    for Filter in filters:
        
        temp=df[df[filter_col]==Filter]
        
        # q3, q1 = np.percentile(temp[mag_col], [75 ,25])
        # iqr=q3-q1
        # h=q3+iqr*1.5
        # l=q1-iqr*1.5
        # temp=temp[(temp[mag_col]<=h)&(temp[mag_col]>=l)]
        
        q99, q1 = np.percentile(temp[mag_col], [99 ,1])
        temp=temp[(temp[mag_col]<=q99)&(temp[mag_col]>=q1)]
        
        
#         q3_err, q1_err = np.percentile(temp['mag_err'], [75 ,25])
#         iqr_err=q3_err-q1_err
#         h_err=q3_err+iqr_err*1.5
            
#         temp=temp[temp['mag_err']<=h_err]
                
        dframe=pd.concat([dframe,temp])
        
        
        
    return dframe

def correct_time(df,mjd_col,ra_col,dec_col,site):
    ip_peg = coord.SkyCoord(df[ra_col].values,df[dec_col].values,
                            unit=(u.deg, u.deg), frame='icrs')
    saao = coord.EarthLocation.of_site(site)
    times = time.Time(list(df[mjd_col]), format='mjd',
                      scale='tt', location=saao) 
    ltt_bary = times.light_travel_time(ip_peg,'barycentric')  
    time_barycentre = times.tdb + ltt_bary
    df.loc[:,'bct']=time_barycentre.jd
    
    return df

def correct_time_asassn(df,mjd_col,ra_col,dec_col):
    ip_peg = coord.SkyCoord(df[ra_col].values,df[dec_col].values,
                            unit=(u.deg, u.deg), frame='icrs')
    saao = coord.EarthLocation.of_site('Cerro Tololo Interamerican Observatory')
    times = time.Time(list(df[mjd_col]), format='jd',
                      scale='utc', location=saao) 
    ltt_bary = times.light_travel_time(ip_peg,'barycentric')  
    time_barycentre = times.tdb + ltt_bary
    df.loc[:,'bct']=time_barycentre.jd
    
    return df


def parse_input(input_str):
    # Check if input_str is a file path
    if input_str.endswith('.txt') or input_str.endswith('.csv'):
        # If it's a file path, read the file and return the values
        values = []
        with open(input_str, 'r') as file:
            if input_str.endswith('.txt'):
                values = [line.strip() for line in file.readlines()]
            elif input_str.endswith('.csv'):
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    values.extend(row)
       
        return values
    else:
        # If it's not a file path, treat it as a single value
        return [input_str]
        
def theta_f_old_version(periods,mag,magerr,mjd):
    theta_array=np.zeros(periods.size)
    
    i=0
    for p in range(periods.size):
        phi=(mjd/periods[p])
        nphi = phi.astype(np.int64)  
        phi = phi - nphi
        idx=np.argsort(phi)
        m=mag[idx]
        merr=magerr[idx]
        w=1/(merr[1:]**2 + merr[:-1]**2)
        theta_array[i]=np.sum(w*(m[1:] - m[:-1])**2)/(np.sum((m[1:]-np.mean(m[1:]))**2)*np.sum(w))
        i+=1
    return theta_array
