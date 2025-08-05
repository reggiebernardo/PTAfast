import numpy as np
from gausspta.pulsar_stats import *


if __name__ == "__main__":

    print(f"Running on {cpu_count()} CPUs"); print()

    # Define configurations for different pulsar counts
    configurations = [
        # {"npsrs": 25, "na_bins": 5},
        # {"npsrs": 50, "na_bins": 10},
        {"npsrs": 100, "na_bins": 15},
    ]

    # Common parameters
    n_sims = 5
    n_bins = 14

    for config in configurations:
        npsrs = config["npsrs"]
        na_bins = config["na_bins"]

        psrs = load_gauss_array(pkl_file=f'my_ideal_pta_{npsrs}')
        noisedict = load_noisedict(pkl_file=f'mynoisedict_{npsrs}')

        # isotropic gwb
        # my_pta_stats = pta_stats_master(
        #     psrs, noisedict, n_sims=n_sims, n_bins=n_bins, na_bins=na_bins,
        #     add_gwb=True, gamma_gw=13/3, log10_A_gw=-15,
        #     add_rn=False, add_wn=False, tmin_mjd=0, save_pickle=True,
        #     pkl_file=f'my_pta_stats_{npsrs}', dir_path='./sims',
        #     show_progress=True
        # )
        
        # kinematic dipole signal
        my_pta_stats = pta_stats_master(
            psrs, noisedict, n_sims=n_sims, n_bins=n_bins, na_bins=na_bins,
            add_gwb=False, gamma_gw=13/3, log10_A_gw=-15,
            add_kin_dipole=True, gamma_kd=13/3, log10_A_kd=-15,
            add_rn=False, add_wn=False, tmin_mjd=0, save_pickle=True,
            pkl_file=f'my_pta_stats_kd_{npsrs}', dir_path='./sims',
            show_progress=True, use_computational_frame=True
        )
