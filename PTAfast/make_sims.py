from . import residuals
from .residuals import Pulsar, make_gauss_array
# from .correlated_noises import add_common_correlated_noise, add_kinematic_dipole

import numpy as np
import os, pickle


def make_ideal_array(npsrs=25, Tobs=15, ntoas=10000, toaerr=1e-7, \
                     isotropic=True, gaps=False, pdist=1., backends='NUPPI.1400', \
                     xz_pulsars=False, \
                     save_pickle=False, pkl_file='my_ideal_pta', \
                     custom_model={'RN':30, 'DM':100, 'Sv':None}, dir_path = './sims'):

    psrs = make_gauss_array(npsrs=npsrs, Tobs=Tobs, ntoas=ntoas, \
                            isotropic=isotropic, gaps=gaps, toaerr=toaerr, \
                            pdist=pdist, backends=backends, \
                            noisedict=None, custom_model=custom_model, \
                            xz_pulsars=xz_pulsars)
    
    for psr in psrs:
        psr.make_ideal()  # flattens out the residuals

    if save_pickle:
        # define the directory path
        # dir_path = './fpta_sims'
        
        # check if the directory exists, and create it if it doesn't
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # save the pickle file
        with open(f'{dir_path}/{pkl_file}.pkl', 'wb') as f:
            pickle.dump(psrs, f)

    print(f'Setup ideal PTA {Tobs}-yr data with {npsrs} pulsars')
    return psrs


def load_gauss_array(pkl_file, dir_path='./sims'):
    psrs = pickle.load(open(f'{dir_path}/{pkl_file}.pkl', 'rb'))

    # get number of pulsars
    npsrs=len(psrs)

    # find the maximum time span
    # tmin = [p.toas.min() for p in psrs]
    # tmax = [p.toas.max() for p in psrs]
    # Tspan = np.max(tmax) - np.min(tmin)
    Tspan = np.amax([psr.toas.max() for psr in psrs]) - np.amin([psr.toas.min() for psr in psrs])
    # Tspan_mjd = Tspan/86400
    Tspan_yr = Tspan/86400/365.25

    print(f'Loaded PTA {int(np.round(Tspan_yr,0))}-yr data with {int(npsrs)} pulsars')
    return psrs


def make_noise_dict(psrs, save_pickle=False, pkl_file='mynoisedict', \
                    dir_path='./sims', backends='NUPPI.1400'): 
    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)
    noisedict={}
    for psr in psrs:
        psrname=psr.name
        # rn_dict[psr.name]={'log10_A': np.random.uniform(-16,-13), \
        #                'gamma': np.random.uniform(2, 6)}
        
        # generate new noise dictionary
        efac = np.random.uniform(0.8, 1.1)
        equad = np.random.uniform(-8, -6)
        ecorr = np.random.uniform(-8, -6)
        RN_Amp = np.random.uniform(-16, -13)
        RN_gamma = np.random.uniform(2, 6)

        # update the dictionary with new values for each pulsar
        noisedict.update({
            f"{psrname}_{backends}_efac": efac,
            f"{psrname}_{backends}_log10_tnequad": equad,
            f"{psrname}_{backends}_log10_ecorr": ecorr,
            f"{psrname}_{backends}_log10_RN_Amp": RN_Amp,
            f"{psrname}_{backends}_RN_gamma": RN_gamma
        })

    if save_pickle:
        # check if the directory exists, and create it if it doesn't
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        with open(f'{dir_path}/{pkl_file}.pkl', 'wb') as f:
            pickle.dump(noisedict, f) # save the pickle file

    return noisedict


def load_noisedict(pkl_file, dir_path='./sims'):
    noisedict = pickle.load(open(f'{dir_path}/{pkl_file}.pkl', 'rb'))

    # count number of pulsars
    pulsar_names = set()
    for key in noisedict.keys():
        # split key by "_" and take the first part as the pulsar name
        pulsar_name = key.split("_")[0]
        pulsar_names.add(pulsar_name)
    # count of unique pulsars
    npsrs = len(pulsar_names)

    print(f'Loaded noise dictionary for {npsrs} pulsars')
    return noisedict

