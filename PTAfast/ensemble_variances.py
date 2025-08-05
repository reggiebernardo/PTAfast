import numpy as np
from .transfer_functions import TF_Alpha, TF_Beta, TF_AlphaBeta, TF_BetaAlpha, Ca0a0, Ca0ak, Ca0bk

# Usage
# k_gauss, ak2_gauss, bk2_gauss=xy_gaussian_ev_master(S=lambda x: powerlaw(x, log10_A=-15, gamma=13/3), \
#                                                     fgw_min=1e-9, fgw_max=1e-7, fgw_N=10000, T_yr=15, k_max=14)


def xy_gaussian_ev(S, fgw_min=1e-9, fgw_max=1e-7, fgw_N=10000, k=1, T_yr=15):
    logfgws = np.linspace(np.log(fgw_min), np.log(fgw_max), fgw_N)
    fgws=np.exp(logfgws)

    dlogfgws = (np.log(fgw_max) - np.log(fgw_min)) / fgw_N
    dfgws=fgws*dlogfgws

    yr_to_sec=3.1536e7
    T_sec=T_yr*yr_to_sec

    akak = np.sum(dfgws*TF_Alpha(fgws, k, k, T_sec)*S(fgws))
    bkbk = np.sum(dfgws*TF_Beta(fgws, k, k, T_sec)*S(fgws))
    results = {'akak': akak, 'bkbk': bkbk}
    return results

def xy_gaussian_ev_dict(S, fgw_min=1e-9, fgw_max=1e-7, fgw_N=10000, T_yr=15, k_max=14):
    k_bins={}
    for k in np.arange(1,k_max+1):
        data=xy_gaussian_ev(S, fgw_min=fgw_min, fgw_max=fgw_max, fgw_N=fgw_N, T_yr=T_yr, k=k)
        data['k']=k; k_bins['bin_'+str(int(k))]=data
    return k_bins

def xy_gaussian_ev_master(S, fgw_min=1e-9, fgw_max=1e-7, fgw_N=10000, T_yr=15, k_max=14):
    xy_ev_dict=xy_gaussian_ev_dict(S, fgw_min=fgw_min, fgw_max=fgw_max, fgw_N=fgw_N, T_yr=T_yr, k_max=k_max)
    k_all, ak2_all, bk2_all = [], [], []
    for vals in xy_ev_dict.values():
        k_all.append(vals['k'])
        ak2_all.append(vals['akak'])  # No need for [0]
        bk2_all.append(vals['bkbk'])  # No need for [0]

    # Convert lists to NumPy arrays
    k_all = np.array(k_all); ak2_all = np.array(ak2_all); bk2_all = np.array(bk2_all)
    return k_all, ak2_all, bk2_all

