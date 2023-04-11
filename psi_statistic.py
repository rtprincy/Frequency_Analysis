#!/usr/bin/env python
# coding: utf-8

from astropy.timeseries import LombScargle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import numba
from astropy.io import fits, ascii
from astropy.table import Table, vstack
import argparse
import os
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--name_data_file", type=str, help="A file containing unique RA and DEC of the objects ")
parser.add_argument("--name_lc_file", type=str, help="A file containing the lightcurve of the objects ")
parser.add_argument("--directory", type=str, default='.',help="Path to the directory containing the data and lightcurves")
parser.add_argument("--save_to_path", type=str, default="periodograms",help="Path to the directory to store the periodogram.")

parser.add_argument("--oversampling_factor", type=int, default=10, help="Oversampling factor ")
parser.add_argument("--maximum_frequency", type=int, default=None, help="If None, then it is computed automatically")
parser.add_argument("--ra_col_name", type=str, default="RA", help="Name of the RA column")
parser.add_argument("--dec_col_name", type=str, default="DEC", help="Name of the DEC column")

parser.add_argument("--idx_start", type=int, default=0, help="Row index in the data file to start computing the periodogram")
parser.add_argument("--idx_end", type=int, default=-1, help="Row index in the data file to end the computation of the periodogram")

parser.add_argument("--mjd_col", type=str, default="MJD-OBS", help="times column name")
parser.add_argument("--mag_col", type=str, default="MAG_OPT", help="magnitude column name")
parser.add_argument("--magerr_col", type=str, default="MAGERR_OPT", help="magnitude error column name")
parser.add_argument("--filter_col", type=str, default="FILTER", help="Filter column name")
parser.add_argument("--ml_data", type=bool, default=True, help="True if the input files are from MeerLICHT data")
parser.add_argument("--filters", type=str, default="qui", help="Filters to use, e.g., qui")

opt = parser.parse_args()

mjd_col=opt.mjd_col
mag_col=opt.mag_col
magerr_col=opt.magerr_col
filter_col=opt.filter_col
ml_data=opt.ml_data
filters=opt.filters

data_name=opt.name_data_file
data_lc_name=opt.name_lc_file
directory=opt.directory
oversampling_fact=opt.oversampling_factor
max_freq=opt.maximum_frequency
ra_col=opt.ra_col_name
dec_col=opt.dec_col_name
save_to_path=opt.save_to_path
start=opt.idx_start
end=opt.idx_end

if os.path.exists(save_to_path+'lsp')==False:
    os.mkdir(save_to_path+'lsp')
    os.mkdir(save_to_path+'theta')


def theta_f(periods,mag,magerr,mjd):
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
        fn = 0.5 / np.median(np.diff(times)) 
    return np.arange(f0, fn, 1/(tbase * oversampling_factor))
# In[7]:


extension_data_lc=data_lc_name.split(sep='.')[-1]

if extension_data_lc=='fits':
    data=fits.open(directory+data_lc_name,memmap=True)
    data = Table(data[1].data).to_pandas()
    print("lightcurves contains %d elements"%(data.shape[0]))
    
elif extension_data_lc=="csv":
    data=pd.read_csv(directory+data_lc_name)
    print("lightcurves contains %d elements"%(data.shape[0]))
elif extension_data_lc=='txt':
    data=pd.read_table(directory+data_lc_name)
    print("lightcurves contains %d elements"%(data.shape[0]))
else: 
    print("Data should be in fits or csv format only")
    
    
extension_data=data_name.split(sep='.')[-1]

if extension_data=='fits':
    filtered_data=fits.open(directory+data_name,memmap=True)
elif extension_data=="csv":
    filtered_data=pd.read_csv(directory+data_name)
    
else: 
    print("Data should be in fits or csv format only")
        

data[['RA','DEC']]=data[['RA','DEC']].round(5)
filtered_data[[ra_col,dec_col]]=filtered_data[[ra_col,dec_col]].round(5)

if ml_data==True:
    
    data=data[data['QC-FLAG']!='red']
    data=data[(data[mag_col]!=99.0)&(data[mag_col]>0.)]

    filtered_data=filtered_data[filtered_data['FLAGS']==0]
    
passbands=list(filters)
   
print("file contains %d unique elements"%(filtered_data.shape[0]))

theta_jit = numba.jit(nopython=True)(theta_f)
theta_compile=theta_jit(np.array([0.2,0.1]),data[mag_col].values,data[magerr_col].values,
data[mjd_col].values) # Just to compile the code with few periods to save time for the next run

if end==-1:
    end=filtered_data.shape[0]
lsp=np.zeros(1)
theta=np.ones(1)
frequencies=np.zeros(1)
for i in tqdm.tqdm(range(start,end)):
    ra,dec=filtered_data[ra_col].values[i],filtered_data[dec_col].values[i]
   
    df=data[(data['RA']==ra)&(data['DEC']==dec)]

    x, y, dyy = [], [], [] # correponds to mjd, mag, and mag_err

    qmag=df[mag_col].where(df[filter_col]=='q').dropna()
    qrms=compute_rms(qmag)

    for pband in passbands:
        x_ = df[mjd_col].where(df[filter_col]==pband).dropna()
        y_ = df[mag_col].where(df[filter_col]==pband).dropna()
        yerr = df[magerr_col].where(df[filter_col]==pband).dropna()

        if (pband !='q') & (y_.size>0):
            y_ = scale_amplitude(y_,qrms)

        else:
            y_ = y_ - np.mean(y_)

        x.append(x_.tolist())
        y.append(y_.tolist())
        dyy.append(yerr.tolist())

    xmulti = np.hstack(x) 
    ymulti = np.hstack(y) 
    dymulti = np.hstack(dyy) 
    
    if  xmulti.size > 39:
        frequencies=freq_grid(times=xmulti,oversampling_factor=oversampling_fact,fn=max_freq)
        periods=1/frequencies
        lsp = LombScargle(t=xmulti, y=ymulti, dy=dymulti,nterms=1).power(frequency=frequencies, method="fastchi2", normalization="psd")
        theta=theta_jit(periods,ymulti,dymulti,xmulti)


    np.save(save_to_path+'lsp/'+'%s_%s_%s.npy'%(str(ra),str(dec),str(i)),np.vstack([frequencies,lsp]))
    np.save(save_to_path+'theta/'+'%s_%s_%s.npy'%(str(ra),str(dec),str(i)),theta)
