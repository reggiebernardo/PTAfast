import numpy as np
from scipy.stats import moment
from tqdm import tqdm
from multiprocess import Pool, cpu_count
import sys, os, pickle

from .pulsar_fft import get_fft
from .make_sims import *
from .residuals import copy_array
from .correlated_noises import compute_rotation_matrices


def compute_eaeb(psr_a, psr_b, eta_eaeb=1e-10):
    pos_a=psr_b.pos
    pos_b=psr_a.pos
    angle = np.arccos(np.dot(pos_a, pos_b)-eta_eaeb)
    return angle


def bin_res_pairs_in_k(psrs, na_bins=15, n_bins=14, tmin_mjd=0):
    # get the minimum and maximum TOAs for all pulsars
    tmin = [p.toas.min() for p in psrs]
    tmax = [p.toas.max() for p in psrs]
    Tspan = np.max(tmax) - np.min(tmin)
    Tspan_yr = np.round(Tspan / 86400 / 365.25, 1)

    # define angular bins
    angular_bin_edges = np.linspace(0, np.pi, na_bins + 1)
    angular_bins = [[] for _ in range(na_bins + 1)]  # add one for 0-degree bin

    # compute angles between each pair of pulsars and bin them
    for i, psr_a in enumerate(psrs):
        for j, psr_b in enumerate(psrs):
            if i == j:
                angular_bins[0].append((psr_a, psr_b)) 
            elif i < j:
                angle = compute_eaeb(psr_a, psr_b)
                bin_index = np.digitize(angle, angular_bin_edges) # - 1 + 1
                angular_bins[bin_index].append((psr_a, psr_b))

    # prepare the containers for the results
    a0a0_bins = [[] for _ in range(na_bins + 1)]
    akak_bins = [[] for _ in range(na_bins + 1)]
    bkbk_bins = [[] for _ in range(na_bins + 1)]
    akbk_bins = [[] for _ in range(na_bins + 1)]
    zeta_bins = [[] for _ in range(na_bins + 1)]

    # bin each pulsar in frequency within each angular bin
    for bin_idx, angular_bin in enumerate(angular_bins):
        for psr_a, psr_b in angular_bin:
            # frequency binning for psr_a
            a0_psr_a, a_psr_a, b_psr_a, _ = \
                get_fft(tmin_mjd + psr_a.toas/86400, psr_a.residuals_gauss, Tspan_yr=Tspan_yr, n_bins=n_bins, )
            # frequency binning for psr_b
            a0_psr_b, a_psr_b, b_psr_b, _ = \
                get_fft(tmin_mjd + psr_b.toas/86400, psr_b.residuals_gauss, Tspan_yr=Tspan_yr, n_bins=n_bins)

            # collect results
            a0a0_bins[bin_idx].append(a0_psr_a*a0_psr_b)
            akak_bins[bin_idx].append(a_psr_a*a_psr_b)
            bkbk_bins[bin_idx].append(b_psr_a*b_psr_b)
            akbk_bins[bin_idx].append(a_psr_a*b_psr_b + a_psr_b*b_psr_a)
            zeta_bins[bin_idx].append(compute_eaeb(psr_a, psr_b))

    # convert results to numpy arrays
    a0a0_bins = [np.array(a0a0_bin) for a0a0_bin in a0a0_bins]
    akak_bins = [np.array(akak_bin) for akak_bin in akak_bins]
    bkbk_bins = [np.array(bkbk_bin) for bkbk_bin in bkbk_bins]
    akbk_bins = [np.array(akbk_bin) for akbk_bin in akbk_bins]
    zeta_bins = [np.array(zeta_bin) for zeta_bin in zeta_bins]

    # return frequencies, 1 yr^-1 = 31.7 nHz
    yrinv_to_nhz = 31.7
    f0 = (1 / Tspan_yr) * yrinv_to_nhz
    freqs_bins = np.arange(1, n_bins + 1) * f0

    return freqs_bins, zeta_bins, a0a0_bins, akak_bins, bkbk_bins, akbk_bins

def initialize_stats_dict(n_angular_bins, n_freq_bins):
    stats_dict = {'E1': [[] for _ in range(n_angular_bins)],
                  'V2': [[] for _ in range(n_angular_bins)],
                  'S3': [[] for _ in range(n_angular_bins)],
                  'K4': [[] for _ in range(n_angular_bins)]}
    for key in stats_dict:
        stats_dict[key] = [[[] for _ in range(n_freq_bins)] for _ in range(n_angular_bins)]
    return stats_dict

def update_stats(stats_dict, akak_bins):
    for i, akak_bin in enumerate(akak_bins):  # loop over angular separation
        if akak_bin.size == 0:
            continue  # Skip empty bins
        for k in range(akak_bin.shape[1]):  # loop over frequency bins
            E1 = np.mean(akak_bin[:, k])
            V2 = np.var(akak_bin[:, k])
            S3 = moment(akak_bin[:, k], moment=3)
            K4 = moment(akak_bin[:, k], moment=4)
            stats_dict['E1'][i][k].append(E1)
            stats_dict['V2'][i][k].append(V2)
            stats_dict['S3'][i][k].append(S3)
            stats_dict['K4'][i][k].append(K4)


# parallelized version of pulsar pair angular and frequency binning
def bin_pairs(pulsar_indices, psrs, angular_bin_edges):
    i, j = pulsar_indices
    psr_a, psr_b = psrs[i], psrs[j]
    if i == j:
        return (0, (psr_a, psr_b))  # add to 0th angular bin
    elif i < j:
        angle = compute_eaeb(psr_a, psr_b)
        bin_index = np.digitize(angle, angular_bin_edges) # - 1 + 1
        return (bin_index, (psr_a, psr_b))
    else:
        return None

def process_angular_bin(args):
    bin_idx, angular_bin, Tspan_yr, n_bins, tmin_mjd = args
    a0a0_bin, akak_bin, bkbk_bin, akbk_bin, zeta_bin = [], [], [], [], []

    for psr_a, psr_b in angular_bin:
        # frequency binning for psr_a
        a0_psr_a, a_psr_a, b_psr_a, _ = get_fft(tmin_mjd + psr_a.toas/86400, psr_a.residuals_gauss, Tspan_yr=Tspan_yr, n_bins=n_bins)
        # frequency binning for psr_b
        a0_psr_b, a_psr_b, b_psr_b, _ = get_fft(tmin_mjd + psr_b.toas/86400, psr_b.residuals_gauss, Tspan_yr=Tspan_yr, n_bins=n_bins)

        # collect the results
        a0a0_bin.append(a0_psr_a*a0_psr_b)
        akak_bin.append(a_psr_a*a_psr_b)
        bkbk_bin.append(b_psr_a*b_psr_b)
        zeta_bin.append(compute_eaeb(psr_a, psr_b))
        # akbk_bin.append(a_psr_a*b_psr_b + a_psr_b*b_psr_a)
        akbk_bin.append(a_psr_a*b_psr_b)

    return bin_idx, np.array(a0a0_bin), np.array(akak_bin), np.array(bkbk_bin), np.array(akbk_bin), np.array(zeta_bin)

def process_angular_bin_cf(args):
    # angular binning in computational frame
    bin_idx, angular_bin, Tspan_yr, n_bins, tmin_mjd, v_pos = args
    a0a0_bin, akak_bin, bkbk_bin, akbk_bin, zeta_bin = [], [], [], [], []

    for psr_a, psr_b in angular_bin:
        # frequency binning for psr_a
        a0_psr_a, a_psr_a, b_psr_a, _ = get_fft(tmin_mjd + psr_a.toas/86400, psr_a.residuals_gauss, Tspan_yr=Tspan_yr, n_bins=n_bins)
        # frequency binning for psr_b
        a0_psr_b, a_psr_b, b_psr_b, _ = get_fft(tmin_mjd + psr_b.toas/86400, psr_b.residuals_gauss, Tspan_yr=Tspan_yr, n_bins=n_bins)

        # collect the results
        a0a0_bin.append(a0_psr_a*a0_psr_b)
        akak_bin.append(a_psr_a*a_psr_b)
        bkbk_bin.append(b_psr_a*b_psr_b)
        zeta_bin.append(compute_eaeb(psr_a, psr_b))

        # compute dipole direction in the computational frame
        R1, R2 = compute_rotation_matrices(psr_a.pos, psr_b.pos)
        v_pos_cf = R2 @ R1 @ v_pos 
        # v_phi = np.sign(v_pos_cf[1])*np.arccos(v_pos_cf[0]/np.sqrt(v_pos_cf[0]**2 + v_pos_cf[1]**2))
        # v_theta = np.arccos(v_pos_cf[2])
        v_phi=np.arctan2(v_pos_cf[1], v_pos_cf[0])
        v_theta = np.arccos(v_pos_cf[2])
        # akbk_bin.append( (a_psr_a*b_psr_b + a_psr_b*b_psr_a )/( np.sin(v_theta)*np.sin(v_phi) ) )
        akbk_bin.append( (a_psr_a*b_psr_b)/(np.sin(v_theta)*np.sin(v_phi)) )

    return bin_idx, np.array(a0a0_bin), np.array(akak_bin), np.array(bkbk_bin), np.array(akbk_bin), np.array(zeta_bin)

def bin_res_pairs_in_k_mp(psrs, na_bins=15, n_bins=14, tmin_mjd=0, \
                          show_progress=False, use_computational_frame=False, \
                          v_pos=np.array([0, 1, 0])):
    # get the minimum and maximum TOAs for all pulsars
    # tmin = [p.toas.min() for p in psrs]
    # tmax = [p.toas.max() for p in psrs]
    # Tspan = np.max(tmax) - np.min(tmin)
    Tspan = np.max([p.toas.max() for p in psrs]) - np.min([p.toas.min() for p in psrs])
    Tspan_yr = np.round(Tspan / 86400 / 365.25, 1)

    # define angular bins
    angular_bin_edges = np.linspace(0, np.pi, na_bins + 1)
    angular_bins = [[] for _ in range(na_bins + 1)]  # add one for 0th angular bin

    # compute angles between each pair of pulsars and bin them
    pulsar_indices = [(i, j) for i in range(len(psrs)) for j in range(len(psrs)) if i <= j]
    
    if show_progress:
        with Pool(cpu_count()) as pool:
            results = list(tqdm(pool.starmap(bin_pairs, \
                                             [(indices, psrs, angular_bin_edges) for indices in pulsar_indices]), \
                                total=len(pulsar_indices), desc="binning pairs"))
    else:
        with Pool(cpu_count()) as pool:
            results = pool.starmap(bin_pairs, [(indices, psrs, angular_bin_edges) for indices in pulsar_indices])

    for result in results:
        if result is not None:
            bin_index, psr_pair = result
            angular_bins[bin_index].append(psr_pair)

    # prepare the containers for the results
    a0a0_bins = [[] for _ in range(na_bins + 1)]
    akak_bins = [[] for _ in range(na_bins + 1)]
    bkbk_bins = [[] for _ in range(na_bins + 1)]
    akbk_bins = [[] for _ in range(na_bins + 1)]
    zeta_bins = [[] for _ in range(na_bins + 1)]

    # bin each pulsar in frequency within each angular bin
    if use_computational_frame==False:
        args = [(bin_idx, angular_bin, Tspan_yr, n_bins, tmin_mjd) for bin_idx, angular_bin in enumerate(angular_bins)]
        
        if show_progress:
            with Pool(cpu_count()) as pool:
                results = list(tqdm(pool.map(process_angular_bin, args), \
                                    total=len(args), desc="processing angular bins"))
        else:
            with Pool(cpu_count()) as pool:
                results = pool.map(process_angular_bin, args)

    else: # angular binning in computational frame
        args = [(bin_idx, angular_bin, Tspan_yr, n_bins, tmin_mjd, v_pos) for bin_idx, angular_bin in enumerate(angular_bins)]
        
        if show_progress:
            with Pool(cpu_count()) as pool:
                results = list(tqdm(pool.map(process_angular_bin_cf, args), \
                                    total=len(args), desc="processing angular bins (computational frame)"))
        else:
            with Pool(cpu_count()) as pool:
                results = pool.map(process_angular_bin_cf, args)

    for bin_idx, a0a0_bin, akak_bin, bkbk_bin, akbk_bin, zeta_bin in results:
        a0a0_bins[bin_idx] = a0a0_bin
        akak_bins[bin_idx] = akak_bin
        bkbk_bins[bin_idx] = bkbk_bin
        akbk_bins[bin_idx] = akbk_bin
        zeta_bins[bin_idx] = zeta_bin

    # return frequencies, 1 yr^-1 = 31.7 nHz
    yrinv_to_nhz = 31.7
    f0 = (1 / Tspan_yr) * yrinv_to_nhz
    freqs_bins = np.arange(1, n_bins + 1) * f0

    return freqs_bins, zeta_bins, a0a0_bins, akak_bins, bkbk_bins, akbk_bins


# master function
def pta_stats_master(psrs, noisedict, n_sims=10, n_bins=14, na_bins=15, \
                     add_gwb=True, gamma_gw=13/3, log10_A_gw=-15, \
                     add_kin_dipole=False, gamma_kd=13/3, log10_A_kd=-15, \
                     v_pos=np.array([0, np.sqrt(2)/2, np.sqrt(2)/2]), v00=1/np.pi, beta=1e-3, \
                     add_rn=False, add_wn=False, tmin_mjd=0, save_pickle=False, \
                     pkl_file='my_pta_stats', dir_path='./sims', \
                     show_progress=True, use_computational_frame=False):
    
    n_angular_bins = na_bins+1  # add one for 0th angular bin (auto-power)
    n_freq_bins = n_bins

    akak_ensemble_stats = initialize_stats_dict(n_angular_bins, n_freq_bins)
    bkbk_ensemble_stats = initialize_stats_dict(n_angular_bins, n_freq_bins)
    akbk_ensemble_stats = initialize_stats_dict(n_angular_bins, n_freq_bins)

    angles_ensemble=[]; freqs_ensemble=[]

    # loop over simulations
    for i in range(n_sims):
        # redirect stdout and stderr to suppress fakepta output
        devnull = open(os.devnull, 'w')
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            psrs_copy=copy_array(psrs, noisedict)
            for psr in psrs_copy:
                psr.make_ideal()
                if add_wn:
                    psr.add_white_noise()
                if add_rn:
                    psr_noisedict=psr.noisedict
                    log10_A_key = next(k for k in psr_noisedict if k.endswith('log10_RN_Amp'))
                    gamma_key = next(k for k in psr_noisedict if k.endswith('RN_gamma'))
                    psr.add_red_noise(log10_A=psr_noisedict[log10_A_key], gamma=psr_noisedict[gamma_key])
            if add_gwb:
                add_common_correlated_noise(psrs_copy, orf='hd', \
                    spectrum='powerlaw', log10_A=log10_A_gw, gamma=gamma_gw)
            if add_kin_dipole:
                add_kinematic_dipole(psrs_copy, log10_A=log10_A_kd, gamma=gamma_kd, \
                                     v_pos=v_pos, v00=v00, beta=beta)
        finally:
            # reset stdout and stderr to their original values
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        print(f'sim-{i} angular and frequency binning')
        freqs_bins, zeta_bins, _, akak_bins, bkbk_bins, akbk_bins=\
        bin_res_pairs_in_k_mp(psrs_copy, na_bins=na_bins, n_bins=n_bins, \
                              tmin_mjd=tmin_mjd, show_progress=show_progress, \
                              use_computational_frame=use_computational_frame, \
                              v_pos=v_pos)
        print()
        angles_ensemble.append(np.array([np.mean(zeta_bin) for zeta_bin in zeta_bins]))
        freqs_ensemble.append(freqs_bins)
        update_stats(akak_ensemble_stats, akak_bins)
        update_stats(bkbk_ensemble_stats, bkbk_bins)
        update_stats(akbk_ensemble_stats, akbk_bins)

    for key in akak_ensemble_stats:
        akak_ensemble_vals=np.array(akak_ensemble_stats[key])
        akak_ensemble_stats[key] = akak_ensemble_vals

    for key in bkbk_ensemble_stats:
        bkbk_ensemble_vals=np.array(bkbk_ensemble_stats[key])
        bkbk_ensemble_stats[key] = bkbk_ensemble_vals

    for key in akbk_ensemble_stats:
        akbk_ensemble_vals=np.array(akbk_ensemble_stats[key])
        akbk_ensemble_stats[key] = akbk_ensemble_vals

    angles_ensemble=np.array(angles_ensemble)
    freqs_ensemble=np.array(freqs_ensemble)

    # input dictionary
    npsrs=len(psrs)
    Tspan_yr = int(np.round((np.max([p.toas.max() for p in psrs]) - np.min([p.toas.min() for p in psrs]))/86400/365.25, 1))
    ntoas=int(np.mean(np.array([len(p.toas) for p in psrs])))
    input_dict={'npsrs': npsrs, 'Tspan_yr': Tspan_yr, 'ntoas': ntoas, \
                'add_gwb': add_gwb, 'gamma_gw': gamma_gw, 'log10_A_gw': log10_A_gw, \
                'add_kin_dipole': add_kin_dipole, 'gamma_kd': gamma_kd, \
                'log10_A_kd': log10_A_kd, 'v_pos': v_pos, 'v00': v00, 'beta': beta, \
                'add_red': add_rn, 'add_wn': add_wn, 'noisedict': noisedict, \
                'n_bins': n_bins, 'na_bins': na_bins, 'n_sims': n_sims}

    pta_stats_dict={'akak_ensemble_stats': akak_ensemble_stats, \
                    'bkbk_ensemble_stats': bkbk_ensemble_stats, \
                    'akbk_ensemble_stats': akbk_ensemble_stats, \
                    'angles_ensemble': angles_ensemble, \
                    'freqs_ensemble': freqs_ensemble, \
                    'input': input_dict}

    if save_pickle:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(f'{dir_path}/{pkl_file}.pkl', 'wb') as f:
            pickle.dump(pta_stats_dict, f) # save the pickle file
        print('results saved to', f'{dir_path}/{pkl_file}.pkl')

    print(f'computed {Tspan_yr} yr-PTA stats with {npsrs} MSPs and {n_sims} realisations')    
    return pta_stats_dict

def load_pta_stats(pkl_file, dir_path='./sims', print_input=True):
    pickle_loc=os.path.join(dir_path, pkl_file + '.pkl')
    if os.path.exists(pickle_loc):
        with open(pickle_loc, 'rb') as f:
            pta_stats_dict = pickle.load(f)
        if print_input:
            input_dict=pta_stats_dict['input']
            Tspan_yr=input_dict['Tspan_yr']
            npsrs=input_dict['npsrs']
            n_sims=input_dict['n_sims']
            print(f'loaded {Tspan_yr} yr-PTA stats with {npsrs} MSPs and {n_sims} realisations')
        return pta_stats_dict
    else:
        print('pre-computed stats do not exist')
