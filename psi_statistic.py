#!/usr/bin/env python
# coding: utf-8

from astropy.timeseries import LombScargle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import numba
from astropy.io import fits, ascii
from astropy.table import Table, vstack
import argparse
import os
import tqdm
from astropy import time, coordinates as coord, units as u
pd.options.mode.chained_assignment = None
from distutils.util import strtobool
import pwkit.pdm as pdm
import warnings
from astropy.utils.exceptions import AstropyWarning

import csv

from frequency_tools import *

warnings.simplefilter('ignore', category=AstropyWarning)
parser = argparse.ArgumentParser()
parser.add_argument("--source_id", metavar='INPUT',type=str, help="Gaia source id or list of Gaia source ids")
parser.add_argument("--source_id_colname", type=str, default='SOURCE_ID',help="Source id column name")

parser.add_argument("--name_lc_file", type=str, help="A file containing the lightcurve of the objects ")

parser.add_argument("--site", type=str, default='lasilla',help="MeerLICHT (saao) or BlacGEM (lasilla) sites")
parser.add_argument("--directory", type=str, default='.',help="Path to the directory containing the data and lightcurves")
parser.add_argument("--save_to_path", type=str, default="periodograms",help="Path to the directory to store the periodogram.")

parser.add_argument("--oversampling_factor", type=int, default=10, help="Oversampling factor ")
parser.add_argument("--maximum_frequency", type=int, default=None, help="If None, then it is computed automatically")

parser.add_argument("--minimum_frequency", type=float, default=None, help="Initial frequency to start the frequency search. This is done automatically for MeerLICHT data")

parser.add_argument("--idx_start", type=int, default=0, help="Row index in the data file to start computing the periodogram")
parser.add_argument("--idx_end", type=int, default=-1, help="Row index in the data file to end the computation of the periodogram")


parser.add_argument("--ra_col_name", type=str, default="RA", help="Name of the RA column for the unique objects data")
parser.add_argument("--dec_col_name", type=str, default="DEC", help="Name of the DEC column for the unique objects data")
parser.add_argument("--mjd_col", type=str, default="MJD_OBS", help="times column name")
parser.add_argument("--mag_col", type=str, default="MAG_OPT", help="magnitude column name")
parser.add_argument("--magerr_col", type=str, default="MAGERR_OPT", help="magnitude error column name")
parser.add_argument("--filter_col", type=str, default="FILTER", help="Filter column name")

parser.add_argument("--scaling_filters", type=str, default="q", help="The filter used to  scale the magnitude values in other filters. The default is q for MeerLICHT.")

parser.add_argument("--passbands", type=str, default=None, help="Filter bands used in the periodogram calculations. If None, all filters present in the data will be used.")

parser.add_argument("--flag_col", type=str, default="QC_FLAG", help="Name of the flag column for MeerLICHT data. Rows with flag='red' will be removed in the lightcurve data.")

parser.add_argument("--object_col_id",type=str,default='asas_sn_id',help="Column name that contains IDs to cross-match the objects with their lightcurves")

parser.add_argument("--catalog", type=str, default='bg_data', help="True if the input files are from MeerLICHT data")

parser.add_argument("--window_function", type=lambda x: bool(strtobool(x)), default=False, help="Indicate whether to compute the spectral window")

parser.add_argument('--rm_outliers',type=lambda x: bool(strtobool(x)),
                   default=False, help="If true, extreme outliers (uncertainties in the magnitudes) in the lightcurve will be removed")

parser.add_argument('--optimise_frequency',type=lambda x: bool(strtobool(x)),
                   default=False, help="Optimise resulting lomb-scargle dominant frequency")

parser.add_argument("--n_best_freq", type=int, default=10, help="Number of best frequencies to select for frequency fine tuning")

opt = parser.parse_args()

passbands=opt.passbands

if passbands is not None:
    passbands=opt.passbands.split(sep=',')
    
args = parser.parse_args()

source_ids = parse_input(opt.source_id)
data_lc_name=opt.name_lc_file


site=opt.site

ra_col=opt.ra_col_name
dec_col=opt.dec_col_name

mjd_col=opt.mjd_col
mag_col=opt.mag_col
magerr_col=opt.magerr_col
filter_col=opt.filter_col
flag_col=opt.flag_col
source_id_col=opt.source_id_colname
catalog=opt.catalog

filter_scale=opt.scaling_filters
col_id=opt.object_col_id
window_function=opt.window_function

n_best_freq=opt.n_best_freq
min_freq=opt.minimum_frequency
rm_outliers=opt.rm_outliers


directory=opt.directory
oversampling_fact=opt.oversampling_factor
max_freq=opt.maximum_frequency


save_to_path=opt.save_to_path
start=opt.idx_start
end=opt.idx_end

optimise_frequency=opt.optimise_frequency


if os.path.exists(save_to_path+'lsp')==False:
    os.mkdir(save_to_path+'lsp')
    os.mkdir(save_to_path+'theta')


extension_data_lc=data_lc_name.split(sep='.')[-1]

if extension_data_lc=='fits':
    data=fits.open(directory+data_lc_name,memmap=True)
    data = Table(data[1].data).to_pandas()
    # print("lightcurves contains %d elements"%(data.shape[0]))
    
elif extension_data_lc=="csv":
    data=pd.read_csv(directory+data_lc_name)
    # print("lightcurves contains %d elements"%(data.shape[0]))
elif extension_data_lc=='txt':
    data=pd.read_table(directory+data_lc_name)
    # print("lightcurves contains %d elements"%(data.shape[0]))
else: 
    print("Data should be in fits or csv format only")
    
    


if  catalog=='ml_data':
    data=data[data[flag_col]!='red']
    data=data[(data[mag_col]<30)&(data[mag_col]>0.)]

 
print("Number of objects: %d"%(len(source_ids)))
print('Remove outliers: ', rm_outliers)


theta_jit = numba.jit(nopython=True)(theta_f)
theta_compile=theta_jit(np.array([0.2,0.1]),data[mag_col].values,data[magerr_col].values,
data[mjd_col].values) # Just to compile the code with few periods to save time for the next run

             
lsp=np.array([-1])
theta=np.array([-2])
frequencies=np.array([-1])


if end==-1:
    end=len(source_ids)

if len(source_ids)>1:  
    skip_header=1
    source_ids=np.array(source_ids[skip_header:],dtype=int)
else:
    source_ids=np.array(source_ids,dtype=int)

if catalog=='bg_data':
    for source_id in tqdm.tqdm(source_ids[start:end]):

            df=data[data[source_id_col]==source_id]
            df.dropna(inplace=True)
          
            if df[df[filter_col]==filter_scale].shape[0] >= 40:
                
                lc_passbands=set(np.unique(df[filter_col].values))
                    
                if passbands is None or set(passbands).issubset(lc_passbands)==False:
                     passbands=lc_passbands
                        
                if rm_outliers==True:
                   df=remove_outliers(df,passbands,mag_col,filter_col)
                    
                df=correct_time(df,mjd_col,ra_col,dec_col,site)
   
                x, y, dyy = [], [], []

                qmag=df[mag_col][df[filter_col]==filter_scale]
                qrms=compute_rms(qmag)
                
                             
                for band in passbands:
                    
                    temp = df[df['FILTER']==band]
    
                    x_ = temp['bct'].values
                    y_ = temp[mag_col].values
                    yerr = temp[magerr_col].values
                    
                    
                    if (band !=filter_scale) & (y_.size>0):
                        
                        y_ = scale_amplitude(y_,qrms)
                      
                    else:
                        
                        y_ = y_ - np.mean(y_)
                        
                    
                    x.append(x_.tolist())
                    y.append(y_.tolist())
                    dyy.append(yerr.tolist())
    
                
                xmulti = np.hstack(x) 
                ymulti = np.hstack(y) 
                dymulti = np.hstack(dyy)
                frequencies=freq_grid(times=xmulti,\
                                      oversampling_factor=oversampling_fact,\
                                      f0=min_freq,fn=max_freq)
                
                xmulti=xmulti - np.median(xmulti)
                      
                periods=1/frequencies
               
        
                lsp = LombScargle(t=xmulti, y=ymulti, dy=dymulti,nterms=1).power(frequency=frequencies, method="fastchi2", normalization="psd")
                theta=theta_f(periods,ymulti,dymulti,xmulti)
                np.save(save_to_path+'lsp/'+'%s.npy'%(str(source_id)),np.vstack([frequencies,lsp]))
                np.save(save_to_path+'theta/'+'%s.npy'%(str(source_id)),theta)
                if optimise_frequency:
                    lspw = LombScargle(t=xmulti, y=np.ones_like(ymulti), dy=None, nterms=1).power(frequency=frequencies, method="fast")
                    
                    freq_optim=freq_grid_optimise(n_best_freq,oversampling_fact,psi,lspw,frequencies)
                    period_optim=1/freq_optim
                    
                    pdm_optim=pdm.pdm(xmulti,ymulti,dymulti,period_optim,20)
                    print('Optimised period: ',pdm_optim.pmin)
                if window_function:
                     if os.path.exists(save_to_path+'window_function/lsp')==False:
                        os.makedirs(save_to_path+'window_function/lsp')
                     lspw = LombScargle(t=xmulti, y=np.ones_like(ymulti), dy=None, nterms=1).power(frequency=frequencies, method="fast")
                     np.save(save_to_path+'window_function/lsp/'+'%s.npy'%(str(source_id)),np.vstack([frequencies,lspw]))
                    
