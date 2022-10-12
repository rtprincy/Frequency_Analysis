#!/usr/bin/env python
# coding: utf-8

# In[1]:


from astropy.timeseries import LombScargle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import numba
save_to_path='periodogram_files_v2/'

# In[2]:


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


# In[28]:

# Old freq_grid
# def freq_grid(times,oversample_factor=10,f0=None,fn=None):

#     df = 1.0 / (times.max() - times.min())
#     if f0 is None:
#         f0 = df
#     if fn is None:
#         fn = 0.5 / np.median(np.diff(times)) 
#     return np.arange(f0, fn, df / oversample_factor)

#New freq_grid
def freq_grid(times,oversample_factor=10,f0=None,fn=None):
    times=np.sort(times)
    df = 4.0 / (times.max() - times.min())
    if f0 is None:
        f0 = df
    if fn is None:
        fn = 0.5 / np.median(np.diff(times)) 
    return np.arange(f0, 480, df / oversample_factor)

# In[4]:


def compute_rms(x_mag):
    return np.sqrt(sum(x_mag**2)/x_mag.size)


# In[5]:


def scale_amplitude(x_mag,q_rms):
    #scale any magnitudes to have the same amplitude as qmag
    x_rms=compute_rms(x_mag)
    x_scaled=x_mag*(q_rms/x_rms) 
    x_scaled= x_scaled - np.median(x_scaled)
    return x_scaled


# In[7]:


data=pd.read_csv('sdb_filtered_starclass_dpts_ugrqi_v2.csv')
filtered_data=pd.read_csv('sdb_filtered_starclass_dpts_ugrqi_unique_v2.csv')


# In[10]:


filtered_data.columns=['ID', 'RA_IN', 'DEC_IN']


# In[13]:

passbands = ['q','u','i']
# Define ML column's name
mjd_col='MJD-OBS'
mag_col='MAG_OPT'
mag_err_col='MAGERR_OPT'
filter_col='FILTER'
# flag_col='flags' # if any

idx=filtered_data.ID.values # idx from sdb selection (red flag, dp>40, class star>0.8)

theta_jit = numba.jit(nopython=True)(theta_f)
theta_compile=theta_jit(np.array([0.2,0.1]),data['MAG_OPT'].values,data['MAGERR_OPT'].values,
data['MJD-OBS'].values) # Just to compile the code with few periods to save time for the next run

for i,(ra, dec) in enumerate(filtered_data[['RA_IN','DEC_IN']].values):
    print('processing: ',idx[i],' RA: ', ra,' DEC: ',dec)
    df=data[(data['RA_IN']==ra)&(data['DEC_IN']==dec)]
    # df = df[df[flag_col]!=2] # remove bad data
    df=df[df[mag_col]>0] # some data might have negative mag (case of forced_photometry)
    x, y, dyy = [], [], [] # correponds to mjd, mag, and mag_err

    qmag=df[mag_col].where(df[filter_col]=='q').dropna()
    qrms=compute_rms(qmag)
    # print('n-obs in %s: '%('q'), qmag.size)
    for pband in passbands:
        x_ = df[mjd_col].where(df[filter_col]==pband).dropna()
        y_ = df[mag_col].where(df[filter_col]==pband).dropna()
        yerr = df[mag_err_col].where(df[filter_col]==pband).dropna()

        if pband !='q':
            y_ = scale_amplitude(y_,qrms)
#             print('n-obs in %s: '%(pband), y_.size)
        else:
            y_ = y_ - np.median(y_)

        x.append(x_.tolist())
        y.append(y_.tolist())
        dyy.append(yerr.tolist())

    xmulti = np.hstack(x) 
    ymulti = np.hstack(y) 
    dymulti = np.hstack(dyy) 
    
    frequencies=freq_grid(xmulti)
    print('frequency trials: ',frequencies.size)
    periods=1/frequencies
    t_lsp0=time.time()

    lsp = LombScargle(t=xmulti, y=ymulti, dy=dymulti,nterms=1).power(frequency=frequencies, method="fastchi2", normalization="psd")

    t_lsp1=time.time()
  
    t_theta0=time.time()
    
    theta=theta_f(periods,ymulti,dymulti,xmulti)

    # lspw = LombScargle(t=xmulti, y=np.ones_like(ymulti), dy=None,nterms=1).power(frequency=frequencies, method="fast", normalization=None) 
    # compute lspw if interested in the lsp of the window function
    t_theta1=time.time()


    print('Time processing for theta:', t_theta1 - t_theta0)
    print('Time processing for lsp:', t_lsp1 - t_lsp0)

#     psi=(2*lsp)/theta
    # period_est=1/frequencies[np.argmax(psi)] # best period

    np.save(save_to_path+'lsp/'+'%s_%s_%sv2.npy'%(str(ra),str(dec),str(idx[i])),np.vstack([frequencies,lsp]))
    np.save(save_to_path+'theta/'+'%s_%s_%sv2.npy.npy'%(str(ra),str(dec),str(idx[i])),theta)
    # np.save('periodogram_files/oversample_10/lspw_%s_1term.npy'%(elt),np.vstack([frequencies,lspw]))
    # # np.save('periodogram_files/thetaw_%s.npy'%(elt))




