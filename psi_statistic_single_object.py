from astropy.timeseries import LombScargle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from astropy.io import fits, ascii
from astropy.table import Table, vstack
import numba

def theta_lk(periods,mag,magerr,mjd):
    theta_array=np.zeros(periods.size)
    for p in range(periods.size):
        phi=(mjd/periods[p])
        nphi = phi.astype(np.int64)  
        phi = phi - nphi
        idx=np.argsort(phi)
        m=mag[idx]
        merr=magerr[idx]
        w=1/(merr[1:]**2 + merr[:-1]**2)
        theta_array[p]=np.sum(w*(m[1:] - m[:-1])**2)/(np.sum((m[1:]-np.mean(m[1:]))**2)*np.sum(w))
    return theta_array
# Old frequency grid
# def freq_grid(times,oversample_factor=5,f0=None,fn=None):
#     times=np.sort(times)
#     df = 1.0 / (times.max() - times.min())
#     if f0 is None:
#         f0 = df
#     if fn is None:
#         fn = 0.5 / np.median(np.diff(times)) 
#     return np.arange(f0, fn, df / oversample_factor)

def freq_grid(times,oversample_factor=10,f0=None,fn=None):
    times=np.sort(times)
    df = 1 / oversample_factor * (times.max() - times.min())
    if f0 is None:
        f0 = 4.0 / (times.max() - times.min())
    if fn is None:
        fn = 480
    return np.arange(f0, fn, df)
  
def compute_rms(x_mag):
    return np.sqrt(sum(x_mag**2)/x_mag.size)
  
def scale_amplitude(x_mag,q_rms):
    #scale any magnitudes to have the same amplitude as qmag
    x_rms=compute_rms(x_mag)
    x_scaled=x_mag*(q_rms/x_rms) 
    x_scaled= x_scaled - np.median(x_scaled)
    return x_scaled

# Load data file. The data below is specific to MeerLICHT data. 
# You can change it to your own data.
df=pd.read_csv('31112167.csv')

passbands = ['q','u','i'] # define filters to use

# Define data column's name
mjd_col='MJD'
mag_col='Mag_Opt'
mag_err_col='Magerr_Opt'
filter_col='Filter'
flag_col='flags' # if any

df = df[df[flag_col]!=2] # remove bad data
df=df[df[mag_col]>0] # few data might points have negative mag (case of forced_photometry)

# Read mjd, mag,mag_err for each filter, scale mag to qmag amplitude, and stack into one array
x, y, dyy = [], [], [] # correponds to mjd, mag, and mag_err

qmag=df[mag_col].where(df[filter_col]=='q').dropna()
qrms=compute_rms(qmag)
print('n-obs in %s: '%('q'), qmag.size)
for pband in passbands:
    x_ = df[mjd_col].where(df[filter_col]==pband).dropna()
    y_ = df[mag_col].where(df[filter_col]==pband).dropna()
    yerr = df[mag_err_col].where(df[filter_col]==pband).dropna()
    
    if pband !='q':
        y_ = scale_amplitude(y_,qrms)
        print('n-obs in %s: '%(pband), y_.size)
    else:
        y_ = y_ - np.median(y_)
    
    x.append(x_.tolist())
    y.append(y_.tolist())
    dyy.append(yerr.tolist())
    
xmulti = np.hstack(x) 
ymulti = np.hstack(y) 
dymulti = np.hstack(dyy) 


frequencies=freq_grid(xmulti) # Define frequency space

periods=1/frequencies

print('Number of trial frequencies: ', frequencies.size)

# No python version for theta_kl for faster run
theta_jit = numba.jit(nopython=True)(theta_lk) 

# Compute Lomb-Scargle periodogram
t_lsp0=time.time()

#You can play with the number of terms nterms below. The higher its value the longer the computation time.
lsp = LombScargle(t=xmulti, y=ymulti, dy=dymulti,nterms=1).power(frequency=frequencies, method="fastchi2", normalization="psd")
# # lspw = LombScargle(t=xmulti, y=np.ones_like(ymulti), dy=None,nterms=1).power(frequency=frequencies, method="fast", normalization=None) 
# # compute lspw if interested in the lsp of the window function
t_lsp1=time.time()

# Compute LK statistic Theta
t_theta0=time.time()

theta=theta_jit(periods,ymulti,dymulti,xmulti)

t_theta1=time.time()


print('Time processing for theta:', t_theta1 - t_theta0)
print('Time processing for lsp:', t_lsp1 - t_lsp0)

# Define Psi statistic 
psi=(2*lsp)/theta
psi_norm=psi/psi.max() 
period_est=1/frequencies[np.argmax(psi)] # best period


# Uncomment below if saving the theta and lsp is necessary
# np.save('periodogram_files/oversample_10/lsp_%s_1term.npy'%(elt),np.vstack([frequencies,lsp]))
# np.save('periodogram_files/oversample_10/theta_%s_1term.npy'%(elt),np.vstack([frequencies,theta]))
# np.save('periodogram_files/oversample_10/lspw_%s_1term.npy'%(elt),np.vstack([frequencies,lspw]))


# Phase folding
# Read original unscaled data to plot the phase lightcurve

qmjd=df[mjd_col].where(df[filter_col]=='q').dropna()
umjd=df[mjd_col].where(df[filter_col]=='u').dropna()
imjd=df[mjd_col].where(df[filter_col]=='i').dropna()

umag=df[mag_col].where(df[filter_col]=='u').dropna()
imag=df[mag_col].where(df[filter_col]=='i').dropna()
#qmag already defined above


qphase=(qmjd/period_est.round(5))%1
uphase=(umjd/period_est.round(5))%1
iphase=(imjd/period_est.round(5))%1


# Simple phase plot. You can still optimize and improve it.
plt.figure(figsize=[8,12])

plt.subplot(211)
plt.title('Period: %.5f d'%(period_est),fontsize=15)
plt.plot(frequencies,psi_norm)
plt.plot(frequencies[np.argmax(psi_norm)],psi_norm.max(),marker='*')
plt.ylabel(r'Power',fontsize=15)
plt.xlabel('Frequency [c/d]',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.xlim([130,165])

plt.minorticks_on()
plt.tick_params('both', length=5, width=1., which='major', direction='in')	
plt.tick_params('both', length=2, width=1, which='minor', direction='in')


plt.subplot(212)


plt.plot(qphase,qmag,'.',label='q')
plt.plot(uphase,umag,'.',label='u')
plt.plot(iphase,imag,'.',label='i')


plt.xlabel('phase',fontsize=15)
plt.ylabel('magnitude',fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.ylim([13.5,15.7])
plt.gca().invert_yaxis()

plt.minorticks_on()
plt.tick_params('both', length=12, width=1., which='major', direction='in')	
plt.tick_params('both', length=2, width=1, which='minor', direction='in')

plt.legend(loc='best')
plt.tight_layout(pad=2)

# plt.savefig('periodogram_2867_3.pdf',format='pdf')
# plt.close()
