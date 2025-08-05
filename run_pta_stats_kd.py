
import numpy as np

from PTAfast.residuals import make_gauss_array, copy_array
from PTAfast.correlated_noises import add_common_correlated_noise, add_kinematic_dipole_a, add_kinematic_dipole, compute_rotation_matrices, gammaI00, gammaI10, gammaI11, gammaV11, kin_dipole_orf_I, kin_dipole_orf_V

from PTAfast.pulsar_fft import pta_fft
from PTAfast.ensemble_variances import xy_gaussian_ev_master
from PTAfast.spectrum import powerlaw

import pickle
from PTAfast.make_sims import *
from PTAfast.pulsar_stats import *
from PTAfast.pulsar_class import *

# for theoretical modelling
from PTAfast.transfer_functions import TF_Alpha, TF_Beta, TF_AlphaBeta, TF_BetaAlpha



if __name__ == "__main__":

    npsrs=100
    psrs=load_gauss_array(pkl_file=f'my_ideal_pta_{npsrs}')
    noisedict=load_noisedict(pkl_file=f'mynoisedict_{npsrs}')

    psrs_copy=copy_array(psrs, noisedict)

    # dipole parameters
    v_pos=np.array([0, 1, 0]); v00=1; beta=1e-1

    # gwb parameters
    log10_A_gw=-15; gamma_gw=13/3


    # generate realizations
    n_sims = 10000; n_bins = 14; tmin_mjd = 0

    angles_ensemble = []
    akak_ensemble = []; bkbk_ensemble = []; akbk_ensemble = []

    for sim in range(n_sims):
        # print progress every 10% of n_sims
        if (sim + 1) % (n_sims // 10) == 0 or sim == 0:
            percent_done = int(100 * (sim + 1) / n_sims)
            print(f"{percent_done} percent of {n_sims} simulations done")


        # inject kinematic dipole signal
        for psr in psrs_copy:
            psr.make_ideal()    
        add_kinematic_dipole(psrs_copy, log10_A=log10_A_gw, gamma=gamma_gw, \
                             v00=v00, beta=beta, v_pos=v_pos)
        

        # recover standard gwb signal + kinematic dipole signal
        # get the minimum and maximum TOAs for all pulsars
        tmin = [p.toas.min() for p in psrs_copy]; tmax = [p.toas.max() for p in psrs_copy]
        Tspan = np.max(tmax) - np.min(tmin); Tspan_yr = np.round(Tspan / 86400 / 365.25, 1)

        angles=[]; akak_bin=[]; bkbk_bin=[]; akbk_bin=[]

        # compute angles between each pair of pulsars and bin them
        # for a, psr_a in enumerate(psrs_copy):
        a=0 # psr_a_idx
        psr_a=psrs_copy[a]
        # frequency binning for psr_a
        _, a_psr_a, b_psr_a, _ = \
            get_fft(tmin_mjd + psr_a.toas/86400, psr_a.residuals_gauss, \
                    Tspan_yr=Tspan_yr, n_bins=n_bins)
        # print(a_psr_a[0], b_psr_a[0])

        for b, psr_b in enumerate(psrs_copy):
            if b!=a:
                # frequency binning for psr_b
                _, a_psr_b, b_psr_b, _ = \
                    get_fft(tmin_mjd + psr_b.toas/86400, psr_b.residuals_gauss, \
                            Tspan_yr=Tspan_yr, n_bins=n_bins)
                # print(a_psr_b[0], b_psr_b[0])

                angle = compute_eaeb(psr_a, psr_b); angles.append(angle)
                akak_bin.append( (a_psr_a*a_psr_b) ); bkbk_bin.append( (b_psr_a*b_psr_b) )

                # compute dipole direction in the computational frame
                R1, R2 = compute_rotation_matrices(psr_a.pos, psr_b.pos)
                v_pos_cf = R2 @ R1 @ v_pos 
                v_phi=np.arctan2(v_pos_cf[1], v_pos_cf[0])
                v_theta = np.arccos(v_pos_cf[2])

                akbk_bin.append( (a_psr_a*b_psr_b)/(np.sin(v_theta)*np.sin(v_phi)) )

        angles = np.array(angles)
        akak_bin = np.array(akak_bin); bkbk_bin = np.array(bkbk_bin)
        akbk_bin = np.array(akbk_bin)

        angles_ensemble.append(angles)
        akak_ensemble.append(akak_bin); bkbk_ensemble.append(bkbk_bin)
        akbk_ensemble.append(akbk_bin)

    angles_ensemble = np.array(angles_ensemble)
    akak_ensemble = np.array(akak_ensemble); bkbk_ensemble = np.array(bkbk_ensemble)
    akbk_ensemble = np.array(akbk_ensemble)


    npsrs = len(psrs_copy)
    pkl_name = f'my_pta_stats_kd_{npsrs}_00000'
    dir='./sims/'
    with open(dir + pkl_name + '.pkl', 'wb') as fp:
        pickle.dump({'angles_ensemble': angles_ensemble, \
                     'akak_ensemble': akak_ensemble, \
                     'bkbk_ensemble': bkbk_ensemble, \
                     'akbk_ensemble': akbk_ensemble, \
                     'input': {'npsrs': npsrs, 'log10A_gw': log10_A_gw, 'gamma_gw': gamma_gw, \
                               'v00': v00, 'v_pos': v_pos, 'beta': beta, \
                               'n_sims': n_sims, 'n_bins': n_bins, 'tmin_mjd': tmin_mjd}}, fp)
    print(f'Saved ensemble pickle to ./data/{pkl_name}.pkl')



