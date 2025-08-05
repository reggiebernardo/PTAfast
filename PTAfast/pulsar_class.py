import numpy as np
from .pulsar_stats import *


# 1 testing gaussianity codes

def C_hd(zeta):
    eps=1e-50
    x=np.cos(zeta)
    y=(1-x)/2
    return (1/2)-(y/4)+(3/2)*y*np.log(y + eps)

def Erarb_gwb(zeta):
    cab = C_hd(zeta)
    caa = 1

    if isinstance(zeta, np.ndarray):
        result = np.where(zeta < 1e-10, caa, cab)
    else:
        if zeta < 1e-2:
            result = caa
        else:
            result = cab
            
    return result

def Vrarb_gwb(zeta):
    cab=Erarb_gwb(zeta)
    return 1 + cab**2

# non variance normalized moments
def S3rarb_gwb(zeta):
    cab=Erarb_gwb(zeta)
    return 2*cab*(3 + cab**2)

def K4rarb_gwb(zeta):
    cab=Erarb_gwb(zeta)
    return 3*(3 + 14*(cab**2) + 3*(cab**4))

# gaussian noise model (uncorrelated)

def Erarb_noise(zeta):
    cab = 0
    caa = 1
    eps_zeta=1e-10
    if isinstance(zeta, np.ndarray):
        result = np.where(zeta < eps_zeta, caa, cab)
    else:
        if zeta < eps_zeta:
            result = caa
        else:
            result = cab
    return result

def Vrarb_noise(zeta):
    Eab=Erarb_noise(zeta)
    return 1 + Eab**2

# non variance normalized moments
def S3rarb_noise(zeta):
    Eab=Erarb_noise(zeta)
    return 2*Eab*(3 + Eab**2)

def K4rarb_noise(zeta):
    Eab=Erarb_noise(zeta)
    return 3*(3 + 14*(Eab**2) + 3*(Eab**4))


# 2 PTAStats class
class PTAStats:
    def __init__(self, pkl_file, dir_path='./sims'):
        self.name = pkl_file
        self.dir_path = dir_path
        # self.log10_ra2_gw_min = log10_ra2_gw_min
        # self.log10_ra2_gw_max = log10_ra2_gw_max

    def load_data(self, print_input=True):
        pta_stats=load_pta_stats(self.name, dir_path='./sims', print_input=print_input)
        self.pta_stats=pta_stats
        # return pta_stats

    def plot_2p_stats(self, ax, k_index=0, res2_unit=1, \
                      color='red', fmt='.', markersize=1, alpha=0.3, \
                      elinewidth=1, capsize=1):
        stats_keys=['E1', 'V2', 'S3', 'K4']
        ensemble_stats_keys=['akak_ensemble_stats', 'bkbk_ensemble_stats']
        pta_stats=self.pta_stats
        for row, stats_key in enumerate(stats_keys):
            for col, ensemble_stats_key in enumerate(ensemble_stats_keys):
                for i in np.arange(1, pta_stats['input']['na_bins']):
                    zeta_ensemble_i = pta_stats['angles_ensemble'][:, i]
                    stats_ensemble_i = pta_stats[ensemble_stats_key][stats_key][i, k_index, :]
                    ax[row, col].errorbar(np.mean(zeta_ensemble_i) * 180 / np.pi, \
                                          xerr=np.std(zeta_ensemble_i) * 180 / np.pi, \
                                          y=np.mean(stats_ensemble_i)/(res2_unit**(row+1)), \
                                          yerr=np.std(stats_ensemble_i)/(res2_unit**(row+1)), \
                                          color=color, fmt=fmt, markersize=markersize, alpha=alpha, \
                                          ecolor=color, elinewidth=elinewidth, capsize=capsize)

    def plot_2p_gaussian(self, ax, ra2, rb2, res2_unit=1, ls='-', lw=1, \
                         color='blue', alpha=0.5, label=None, \
                         Erarb=Erarb_gwb, Vrarb=Vrarb_gwb, S3rarb=S3rarb_gwb, K4rarb=K4rarb_gwb):
        
        zta_vals=np.logspace(-3, np.log10(np.pi), 1000)
        for i in range(2):
            if i==0:
                rk2=ra2
                ax[0,i].plot(zta_vals*180/np.pi, rk2*Erarb(zta_vals)/res2_unit, \
                            ls=ls, lw=lw, color=color, alpha=alpha, label=label)
                ax[1,i].plot(zta_vals*180/np.pi, (rk2**2)*Vrarb(zta_vals)/(res2_unit**2), \
                            ls=ls, lw=lw, color=color, alpha=alpha, label=label)
                ax[2,i].plot(zta_vals*180/np.pi, (rk2**3)*S3rarb(zta_vals)/(res2_unit**3), \
                            ls=ls, lw=lw, color=color, alpha=alpha, label=label)
                ax[3,i].plot(zta_vals*180/np.pi, (rk2**4)*K4rarb(zta_vals)/(res2_unit**4), \
                            ls=ls, lw=lw, color=color, alpha=alpha, label=label)
            if i==1:
                rk2=rb2
                ax[0,i].plot(zta_vals*180/np.pi, rk2*Erarb(zta_vals)/res2_unit, \
                            ls=ls, lw=lw, color=color, alpha=alpha, label=label)
                ax[1,i].plot(zta_vals*180/np.pi, (rk2**2)*Vrarb(zta_vals)/(res2_unit**2), \
                            ls=ls, lw=lw, color=color, alpha=alpha, label=label)
                ax[2,i].plot(zta_vals*180/np.pi, (rk2**3)*S3rarb(zta_vals)/(res2_unit**3), \
                            ls=ls, lw=lw, color=color, alpha=alpha, label=label)
                ax[3,i].plot(zta_vals*180/np.pi, (rk2**4)*K4rarb(zta_vals)/(res2_unit**4), \
                            ls=ls, lw=lw, color=color, alpha=alpha, label=label)

    def get_rkrk_stats(self, stats_key='E1', ensemble_stats_key='akak_ensemble_stats', k_index=0):
        '''
        stats_key='E1', 'V2', 'S3', 'K4'
        ensemble_stats_key='akak_ensemble_stats', 'bkbk_ensemble_stats'
        '''
        pta_stats=self.pta_stats
        zeta_stats=[] # angular bins
        rkrk_stats=[]
        for i in np.arange(1, pta_stats['input']['na_bins']):
            zeta_ensemble_i = pta_stats['angles_ensemble'][:, i]
            stats_ensemble_i = pta_stats[ensemble_stats_key][stats_key][i, k_index, :]
            zeta_stats.append(zeta_ensemble_i)
            rkrk_stats.append(stats_ensemble_i)
        return zeta_stats, rkrk_stats

    def plot_rkrk_density(self, ax, stats_key='E1', ensemble_stats_key='akak_ensemble_stats', \
                          k_index=0, res2_unit=1e-12, zinds=[0,2,12,13], \
                          bins=15, histtype='step', ls='-', density=True, alpha=0.8):
        zeta_stats, rkrk_stats=self.get_rkrk_stats(stats_key=stats_key, \
                                                   ensemble_stats_key=ensemble_stats_key, \
                                                   k_index=k_index)
        stats_keys=['E1', 'V2', 'S3', 'K4']
        stats_key_idx=stats_keys.index(stats_key)
        line_widths=np.arange(0.5, 0.5*(len(zinds)+1), 0.5)
        for idx, (i, lw) in enumerate(zip(zinds, line_widths)):
            rkrk_stat = rkrk_stats[i]
            ax.hist(rkrk_stat/(res2_unit**(stats_key_idx+1)), \
                    bins=bins, histtype=histtype, lw=lw, ls=ls, density=density, alpha=0.8, \
                    label=fr'$\zeta\sim${int(np.mean(zeta_stats[i])*180/np.pi)}$^\circ$')