import numpy as np
from scipy.interpolate import pchip_interpolate #, interp1d


# def get_fft(t, r, Tspan_yr=30, n_bins=30, xx=10):
#     '''compute Fourier components of a time series (t, r)'''
#     mask = np.isfinite(r)
#     ts=t-np.min(t) # corrects for phase in fft calculation
#     interp_func = interp1d(ts[mask], r[mask], kind='linear', fill_value="extrapolate")
#     t_interp = np.linspace(0, (Tspan_yr*365.25), len(ts) * xx)
#     r_interp = interp_func(t_interp)
    
#     # fft calculation
#     N = len(t_interp)
#     a0_fft=np.sum(r_interp)*2/N
#     a_fft=np.array([np.sum(r_interp*np.sin(2*np.pi*k*np.arange(N)/N)) for k in np.arange(1, n_bins+1)])*2/N
#     b_fft=np.array([np.sum(r_interp*np.cos(2*np.pi*k*np.arange(N)/N)) for k in np.arange(1, n_bins+1)])*2/N

#     # return frequencies, 1 yr^-1 = 31.7 nHz
#     yrinv_to_nhz = 31.7
#     f0=(1/Tspan_yr)*yrinv_to_nhz
#     freqs=np.arange(1, n_bins+1)*f0
#     return a0_fft, a_fft, b_fft, freqs

def get_fft(t, r, Tspan_yr=30, n_bins=30, xx=10):
    '''Compute Fourier components of a time series (t, r) with improved interpolation handling.'''
    
    # ensure finite values
    mask = np.isfinite(r)
    ts = t[mask] - np.min(t[mask])  # corrects for phase in FFT calculation
    r = r[mask]
    
    # ensure unique time values for interpolation
    unique_t, unique_indices = np.unique(ts, return_index=True)
    unique_r = r[unique_indices]
    
    # eeplace nans/infs in residuals with the median
    unique_r = np.where(np.isfinite(unique_r), unique_r, np.nanmedian(unique_r))
    
    # interpolation using pchip for robustness
    t_interp = np.linspace(0, (Tspan_yr*365.25), len(ts)*xx)
    r_interp = pchip_interpolate(unique_t, unique_r, t_interp)
    
    # FFT calculation
    N = len(t_interp)
    a0_fft = np.sum(r_interp)*2/N
    a_fft=np.array([np.sum(r_interp*np.sin(2*np.pi*k*np.arange(N)/N)) for k in np.arange(1, n_bins + 1)])*2/N
    b_fft=np.array([np.sum(r_interp*np.cos(2*np.pi*k*np.arange(N)/N)) for k in np.arange(1, n_bins + 1)])*2/N
    
    # return frequencies, 1 yr^-1 = 31.7 nHz
    yrinv_to_nhz = 31.7
    f0 = (1 / Tspan_yr)*yrinv_to_nhz
    freqs = np.arange(1, n_bins + 1)*f0
    
    return a0_fft, a_fft, b_fft, freqs


def bin_res_in_k(psrs, n_bins=30, tmin_mjd=0, xx=10, res_gauss=True):
    '''bin pulsar residuals in fourier space'''
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan=np.max(tmax)-np.min(tmin)
    Tspan_yr=np.round(Tspan/86400/365.25,1)

    a0_bins = []
    a_bins = []
    b_bins = []
    freqs_bins = []
    pos_bins = []

    for psr in psrs:
        if res_gauss:
            a0_psr, a_psr, b_psr, freqs_psr=\
            get_fft(tmin_mjd+psr.toas/86400, psr.residuals_gauss, \
                    Tspan_yr=Tspan_yr, n_bins=n_bins, xx=xx)
        else:
            a0_psr, a_psr, b_psr, freqs_psr=\
            get_fft(tmin_mjd+psr.toas/86400, psr.residuals, \
                    Tspan_yr=Tspan_yr, n_bins=n_bins, xx=xx)
        a0_bins.append(a0_psr)
        a_bins.append(a_psr)
        b_bins.append(b_psr)
        freqs_bins.append(freqs_psr)
        pos_bins.append(psr.pos)

    a0_bins=np.array(a0_bins)
    a_bins=np.array(a_bins)
    b_bins=np.array(b_bins)
    freqs_bins=np.array(freqs_bins)
    pos_bins=np.array(pos_bins)
    
    return a0_bins, a_bins, b_bins, freqs_bins, pos_bins


def pta_fft(psrs, n_bins=14, xx=10, res_gauss=True):
    _, a_t, b_t, f_t, _=bin_res_in_k(psrs,n_bins=n_bins, xx=xx, res_gauss=res_gauss)
    freqs=np.mean(f_t,axis=0)
    var_a=[]; var_b=[]; ave_ab=[]
    for i in range(len(freqs)): # compute sample variances
        var_a.append(np.var(a_t[:, i])); var_b.append(np.var(b_t[:, i]))
        ave_ab.append(np.mean( a_t[:, i]*b_t[:, i] ))

    return freqs, np.array(var_a), np.array(var_b), np.array(ave_ab)

