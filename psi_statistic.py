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
from astropy import time, coordinates as coord, units as u
pd.options.mode.chained_assignment = None
from distutils.util import strtobool



parser = argparse.ArgumentParser()
parser.add_argument("--name_data_file", type=str, help="A file containing unique RA and DEC of the objects ")
parser.add_argument("--name_lc_file", type=str, help="A file containing the lightcurve of the objects ")
parser.add_argument("--directory", type=str, default='.',help="Path to the directory containing the data and lightcurves")
parser.add_argument("--save_to_path", type=str, default="periodograms",help="Path to the directory to store the periodogram.")
parser.add_argument("--oversampling_factor", type=int, default=10, help="Oversampling factor ")
parser.add_argument("--maximum_frequency", type=int, default=None, help="If None, then it is computed automatically")
parser.add_argument("--minimum_frequency", type=float, default=0.05, help="Initial frequency to start the frequency search. This is done automatically for MeerLICHT data")
parser.add_argument("--ra_col_name1", type=str, default="RA", help="Name of the RA column for the unique objects data")
parser.add_argument("--dec_col_name1", type=str, default="DEC", help="Name of the DEC column for the unique objects data")
parser.add_argument("--ra_col_name2", type=str, default="RA", help="Name of the RA column for the lightcurve data")
parser.add_argument("--dec_col_name2", type=str, default="DEC", help="Name of the DEC column for the lightcurve data")
parser.add_argument("--idx_start", type=int, default=0, help="Row index in the data file to start computing the periodogram")
parser.add_argument("--idx_end", type=int, default=-1, help="Row index in the data file to end the computation of the periodogram")
parser.add_argument("--mjd_col", type=str, default="MJD-OBS", help="times column name")
parser.add_argument("--mag_col", type=str, default="MAG_OPT", help="magnitude column name")
parser.add_argument("--magerr_col", type=str, default="MAGERR_OPT", help="magnitude error column name")
parser.add_argument("--filter_col", type=str, default="FILTER", help="Filter column name")
parser.add_argument("--filters", type=str, default="qui", help="Filters to use, e.g., qui")
parser.add_argument("--scaling_filters", type=str, default="q", help="The filter used to  scale the magnitude values in other filters. The default is q for MeerLICHT.")
parser.add_argument("--flag_col", type=str, default="QC-FLAG", help="Name of the flag column for MeerLICHT data. Rows with flag='red' will be removed in the lightcurve data.")
parser.add_argument("--object_col_id",type=str,default='asas_sn_id',help="Column name that contains IDs to cross-match the objects with their lightcurves")
parser.add_argument("--ml_data", type=lambda x: bool(strtobool(x)), default=True, help="True if the input files are from MeerLICHT data")
parser.add_argument("--window_function", type=lambda x: bool(strtobool(x)), default=False, help="Indicate whether to compute the spectral window")
parser.add_argument('--remove_outliers',type=lambda x: bool(strtobool(x)),
                   default=False, help="If true, outliers in the lightcurve will be removed")


opt = parser.parse_args()
mjd_col=opt.mjd_col
mag_col=opt.mag_col
magerr_col=opt.magerr_col
filter_col=opt.filter_col
flag_col=opt.flag_col
ml_data=opt.ml_data
filters=opt.filters
filter_scale=opt.scaling_filters
col_id=opt.object_col_id
window_function=opt.window_function

min_freq=opt.minimum_frequency
remove_outliers=opt.remove_outliers
data_name=opt.name_data_file
data_lc_name=opt.name_lc_file
directory=opt.directory
oversampling_fact=opt.oversampling_factor
max_freq=opt.maximum_frequency
ra_col1=opt.ra_col_name1
dec_col1=opt.dec_col_name1

ra_col2=opt.ra_col_name2
dec_col2=opt.dec_col_name2

save_to_path=opt.save_to_path
start=opt.idx_start
end=opt.idx_end

cols=[ra_col2,dec_col2,mjd_col,mag_col,magerr_col,filter_col,flag_col]

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

def remove_outliers(df,filters,mag_col,filter_col):
    dframe=pd.DataFrame()
    for Filter in filters:
        q3, q1 = np.percentile(df[mag_col][df[filter_col]==Filter], [75 ,25])
        iqr=q3-q1
        h=q3+iqr*1.5
        l=q1-iqr*1.5
        mag=df[mag_col][df[filter_col]==Filter]
        mag=mag[(mag<=h)&(mag>=l)]
        df_=df[(df[filter_col]==Filter)&(df[mag_col]<=h)&(df[mag_col]>=l)]
        dframe=pd.concat([dframe,df_])
    return dframe,iqr

def correct_time(df,mjd_col,ra_col,dec_col):
    ip_peg = coord.SkyCoord(df[ra_col].values,df[dec_col].values,
                            unit=(u.deg, u.deg), frame='icrs')
    saao = coord.EarthLocation.of_site('SAAO')
    times = time.Time(list(df[mjd_col]), format='mjd',
                      scale='utc', location=saao) 
    ltt_bary = times.light_travel_time(ip_peg,'barycentric')  
    time_barycentre = times.tdb + ltt_bary
    df.loc[:,'bct']=time_barycentre.value
    
    return df

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
    data_unique=fits.open(directory+data_name,memmap=True)
elif extension_data=="csv":
    data_unique=pd.read_csv(directory+data_name)
    
else: 
    print("Data should be in fits or csv format only")

    


if ml_data==True:
    data=data[cols]
    data[[ra_col2,dec_col2]]=data[[ra_col2,dec_col2]].round(5)
    data_unique[[ra_col1,dec_col1]]=data_unique[[ra_col1,dec_col1]].round(5)

    data=data[data[flag_col]!='red']
    data=data[(data[mag_col]!=99.0)&(data[mag_col]>0.)]
    data_unique=data_unique[data_unique['FLAGS']==0]
    
  
    
passbands=list(filters)
    
    
print("file contains %d unique elements"%(data_unique.shape[0]))


theta_jit = numba.jit(nopython=True)(theta_f)
theta_compile=theta_jit(np.array([0.2,0.1]),data[mag_col].values,data[magerr_col].values,
data[mjd_col].values) # Just to compile the code with few periods to save time for the next run

if end==-1:
    end=data_unique.shape[0]
    
lsp=np.zeros(1)
theta=np.ones(1)
frequencies=np.zeros(1)

if ml_data:
    for i in tqdm.tqdm(range(start,end)):
        ra,dec=data_unique[ra_col1].values[i],data_unique[dec_col1].values[i]

        df=data[(data[ra_col2]==ra)&(data[dec_col2]==dec)]
        # print("lc size: %d"%(df.shape[0]))
        df.dropna(inplace=True)

        df=correct_time(df,mjd_col,ra_col2,dec_col2)

        if remove_outliers==True:
            df=remove_outliers(df,passbands,mag_col,filter_col)

        qmag=df[mag_col][df[filter_col]==filter_scale]
        qrms=compute_rms(qmag)

        x, y, dyy = [], [], [] # correponds to mjd, mag, and mag_err

        for pband in passbands:

            x_ = df['bct'][df[filter_col]==pband]
            y_ = df[mag_col][df[filter_col]==pband]
            yerr = df[magerr_col][df[filter_col]==pband]

            if (pband !=filter_scale) & (y_.size>0):
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
        

else:
    
    obj_id=data_unique[col_id].values
    data[mjd_col]=data[mjd_col] - 2450000 # For ASAS-SN data
    
    for i in tqdm.tqdm(range(start,end)):
        

        df=data[data[col_id]==obj_id[i]]
        # print("lc size: %d"%(df.shape[0]))
        df.dropna(inplace=True)

        if remove_outliers==True:
            df=remove_outliers(df,passbands,mag_col,filter_col)
        
        passbands=np.unique(list(df[filter_col]))
        
        mag=df[mag_col][df[filter_col]==filter_scale]
        rms=compute_rms(mag)

        x, y, dyy = [], [], [] # correponds to mjd, mag, and mag_err

        for pband in passbands:

            x_ = df[mjd_col][df[filter_col]==pband]
            y_ = df[mag_col][df[filter_col]==pband]
            yerr = df[magerr_col][df[filter_col]==pband]

            if (pband !=filter_scale) & (y_.size>0):
                y_ = scale_amplitude(y_,rms)

            else:
                y_ = y_ - np.mean(y_)

            x.append(x_.tolist())
            y.append(y_.tolist())
            dyy.append(yerr.tolist())

        xmulti = np.hstack(x) 
        ymulti = np.hstack(y) 
        dymulti = np.hstack(dyy) 

        if  xmulti.size > 39:
            frequencies=freq_grid(times=xmulti,oversampling_factor=oversampling_fact,
                                  f0=min_freq,fn=max_freq)
            periods=1/frequencies


            lsp = LombScargle(t=xmulti, y=ymulti, dy=dymulti,nterms=1).power(frequency=frequencies, method="fastchi2", normalization="psd")

            theta=theta_jit(periods,ymulti,dymulti,xmulti)


        np.save(save_to_path+'lsp/'+'%s.npy'%(str(obj_id[i])),np.vstack([frequencies,lsp]))
        np.save(save_to_path+'theta/'+'%s.npy'%(str(obj_id[i])),theta)

        
if window_function:
    for i in tqdm.tqdm(range(start,end)):
        ra,dec=data_unique[ra_col1].values[i],data_unique[dec_col1].values[i]

        df=data[(data[ra_col2]==ra)&(data[dec_col2]==dec)]
        # print("lc size: %d"%(df.shape[0]))
        df.dropna(inplace=True)

        df=correct_time(df,mjd_col,ra_col2,dec_col2)

        if remove_outliers==True:
            df=remove_outliers(df,passbands,mag_col,filter_col)

        x, y, dyy = [], [], [] # correponds to mjd, mag, and mag_err

        for pband in passbands:

            x_ = df['bct'][df[filter_col]==pband]
            yerr = df[magerr_col][df[filter_col]==pband]

            x.append(x_.tolist())
            dyy.append(yerr.tolist())

        xmulti = np.hstack(x) 
        ymulti = np.ones(xmulti.size) 
        dymulti = np.hstack(dyy) 

        if  xmulti.size > 39:
            frequencies=freq_grid(times=xmulti,oversampling_factor=oversampling_fact,fn=max_freq)

            periods=1/frequencies

            lsp = LombScargle(t=xmulti, y=np.ones_like(ymulti), dy=None, nterms=1).power(frequency=frequencies, method="fast")

        if os.path.exists(save_to_path+'window_function/lsp')==False:
            os.makedirs(save_to_path+'window_function/lsp')

        np.save(save_to_path+'window_function/lsp/'+'%s_%s_%s.npy'%(str(ra),str(dec),str(i)),np.vstack([frequencies,lsp]))

    
