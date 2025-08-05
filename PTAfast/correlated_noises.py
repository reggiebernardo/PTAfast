from .residuals import Pulsar
from .transfer_functions import TF_Alpha, TF_Beta, TF_AlphaBeta, TF_BetaAlpha, Ca0a0

import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d
import healpy as hp
import importlib, inspect

from .spectrum import powerlaw

# load spectrum functions from "spectrum.py"
module = importlib.import_module('PTAfast.spectrum')
spec = dict(inspect.getmembers(module, inspect.isfunction))

# Misc functions
def get_correlation(psr_a, psr_b, res_a, res_b):

    angle = np.arccos(np.dot(psr_a.pos, psr_b.pos))
    corr = np.dot(res_a, res_b) / len(res_a)

    return corr, angle

def get_correlations(psrs, res):

    corrs = np.array([])
    angles = np.array([])
    autocorrs = np.array([])
    for i in range(len(psrs)):
        for j in range(i+1):
            c, a = get_correlation(psrs[i], psrs[j], res[i], res[j])
            if i == j:
                autocorrs = np.append(autocorrs, c)
            else:
                corrs = np.append(corrs, c)
                angles = np.append(angles, a)
    return corrs, angles, autocorrs

def bin_curve(corrs, angles, bins):

    edges = np.linspace(0., np.pi, bins+1)
    bin_angles = edges[:-1] + 0.5*(edges[1]-edges[0])
    mean = []
    std = []
    for i in range(bins):
        mask = angles > edges[i]
        mask *= angles < edges[i+1]
        mean.append(np.mean(corrs[mask]))
        std.append(np.std(corrs[mask]))
    return np.array(mean), np.array(std), np.array(bin_angles)

# ORFs
def create_gw_antenna_pattern(pos, gwtheta, gwphi):

    m = np.array([np.sin(gwphi), -np.cos(gwphi), np.zeros(len(gwphi))]).T
    n = np.array([-np.cos(gwtheta) * np.cos(gwphi), -np.cos(gwtheta) * np.sin(gwphi), np.sin(gwtheta)]).T
    omhat = np.array([-np.sin(gwtheta) * np.cos(gwphi), -np.sin(gwtheta) * np.sin(gwphi), -np.cos(gwtheta)]).T

    fplus = 0.5 * (np.dot(m, pos) ** 2 - np.dot(n, pos) ** 2) / (1 + np.dot(omhat, pos))
    fcross = (np.dot(m, pos) * np.dot(n, pos)) / (1 + np.dot(omhat, pos))
    cosMu = -np.dot(omhat, pos)

    return fplus, fcross, cosMu

def hd(psrs):
    orfs = np.zeros((len(psrs), len(psrs)))
    for i in range(len(psrs)):
        for j in range(len(psrs)):
            if i == j:
                orfs[i, j] = 1.
            else:
                omc2 = (1 - np.dot(psrs[i].pos, psrs[j].pos)) / 2
                orfs[i, j] =  1.5 * omc2 * np.log(omc2) - 0.25 * omc2 + 0.5
    return orfs

def anisotropic(psrs, h_map):

    orfs = np.zeros((len(psrs), len(psrs)))
    npixels = len(h_map)
    pixels = hp.pix2ang(hp.npix2nside(npixels), np.arange(npixels), nest=False)
    gwtheta = pixels[0]
    gwphi = pixels[1]
    for i in range(len(psrs)):
        for j in range(len(psrs)):
            if i == j:
                k_ab = 2.
            else:
                k_ab = 1.
            fp_a, fc_a, _ = create_gw_antenna_pattern(psrs[i].pos, gwtheta, gwphi)
            fp_b, fc_b, _ = create_gw_antenna_pattern(psrs[j].pos, gwtheta, gwphi)
            orfs[i, j] = 1.5 * k_ab * np.sum((fp_a*fp_b + fc_a*fc_b) * h_map) / npixels
    return orfs

def monopole(psrs):
    npsr = len(psrs)
    return np.ones((npsr, npsr))

def dipole(psrs):
    orfs = np.zeros((len(psrs), len(psrs)))
    for i in range(len(psrs)):
        for j in range(len(psrs)):
            if i == j:
                orfs[i, j] = 1.
            else:
                omc2 = np.dot(psrs[i].pos, psrs[j].pos)
                orfs[i, j] = omc2
    return orfs

def curn(psrs):
    npsr = len(psrs)
    return np.eye(npsr)


# Noise generating function
def add_common_correlated_noise(psrs, orf='hd', spectrum='powerlaw', name='gw', \
                                idx=0, components=50, freqf=1400, custom_psd=None, f_psd=None, h_map=None, \
                                fgw_min=1e-9, fgw_max=1e-7, fgw_N=10000, \
                                add_dc=False, **kwargs):

    if name is not None:
        signal_name = name + '_common'
    else:
        signal_name = 'common'

    Tspan = np.amax([psr.toas.max() for psr in psrs]) - np.amin([psr.toas.min() for psr in psrs])

    # dynamically adjust components to be less than fgw_max; otherwise there'd be artificial high frequency noise
    while (components / Tspan) > fgw_max:
        components -= 1

    # print(components)
    if f_psd is None:
        f_psd = np.arange(1, components+1) / Tspan
    df = np.diff(np.append(0., f_psd))

    # for residuals_gauss
    logfgws = np.linspace(np.log(fgw_min), np.log(fgw_max), fgw_N)
    fgws=np.exp(logfgws)

    dlogfgws = (np.log(fgw_max) - np.log(fgw_min))/fgw_N
    dfgws=fgws*dlogfgws

    # if spectrum is 'custom':
    if spectrum == 'custom':
        # assert f_psd is None, '"f_psd" must not be None. The frequencies "f_psd" correspond to frequencies where the "custom_psd" is evaluated.'
        assert len(custom_psd) == len(f_psd), '"custom_psd" and "f_psd" must be same length. The frequencies "f_psd" correspond to frequencies where the "custom_psd" is evaluated.'
        psd_gwb = custom_psd
        psd_gwb_gauss = custom_psd # for residuals_gauss
    elif spectrum in [*spec]:
        psd_gwb = spec[spectrum](f_psd, **kwargs)
        psd_gwb_gauss = spec[spectrum](fgws, **kwargs) # for residuals_gauss
        for psr in psrs:
            psr.update_noisedict(signal_name, kwargs)

    # save noise properties in signal model
    for psr in psrs:
        if signal_name in [*psr.signal_model]:
            psr.residuals -= psr.reconstruct_signal(signals=[signal_name])

        psr.signal_model[signal_name] = {}
        psr.signal_model[signal_name]['orf'] = orf
        psr.signal_model[signal_name]['spectrum'] = spectrum
        psr.signal_model[signal_name]['hmap'] = h_map
        psr.signal_model[signal_name]['f'] = f_psd
        psr.signal_model[signal_name]['psd'] = psd_gwb
        psr.signal_model[signal_name]['fourier'] = np.vstack((np.zeros(components), np.zeros(components)))
        psr.signal_model[signal_name]['nbin'] = components
        psr.signal_model[signal_name]['idx'] = idx
    
    psd_gwb = np.repeat(psd_gwb, 2)
    coeffs = np.sqrt(psd_gwb)
    orf_funcs = {'hd':hd, 'monopole':monopole, 'dipole':dipole, 'curn':curn}
    if orf in [*orf_funcs]:
        orfs = orf_funcs[orf](psrs)
    elif orf == 'anisotropic':
        orfs = anisotropic(psrs, h_map)

    for i in range(components):
        orf_corr_sin = np.random.multivariate_normal(mean=np.zeros(len(psrs)), cov=orfs)
        orf_corr_cos = np.random.multivariate_normal(mean=np.zeros(len(psrs)), cov=orfs)

        # residuals---same variance
        for n, psr in enumerate(psrs):
            psr.signal_model[signal_name]['fourier'][0, i] = orf_corr_cos[n] * coeffs[2*i] / df[i]**0.5
            psr.signal_model[signal_name]['fourier'][1, i] = orf_corr_sin[n] * coeffs[2*i+1] / df[i]**0.5
            psr.residuals += orf_corr_cos[n] * (freqf/psr.freqs)**idx * df[i]**0.5 * coeffs[2*i] * np.cos(2*np.pi*f_psd[i]*psr.toas)
            psr.residuals += orf_corr_sin[n] * (freqf/psr.freqs)**idx * df[i]**0.5 * coeffs[2*i+1] * np.sin(2*np.pi*f_psd[i]*psr.toas)


        # residuals---with transfer functions
        # <alpha alpha> and <beta beta>
        var_sin = np.sum(dfgws*TF_Alpha(fgws, i+1, i+1, Tspan)*psd_gwb_gauss)
        var_cos = np.sum(dfgws*TF_Beta(fgws, i+1, i+1, Tspan)*psd_gwb_gauss)

        coeffs_sin = np.sqrt(var_sin); coeffs_cos = np.sqrt(var_cos)
        for n, psr in enumerate(psrs):
            psr.residuals_gauss += orf_corr_cos[n]*coeffs_cos*np.cos(2*np.pi*f_psd[i]*psr.toas)
            psr.residuals_gauss += orf_corr_sin[n]*coeffs_sin*np.sin(2*np.pi*f_psd[i]*psr.toas)
            
    if add_dc: # static DC bin
        var_dc=np.sum(dfgws*Ca0a0(fgws, Tspan)*psd_gwb_gauss)
        coeffs_dc = np.sqrt(var_dc)
        orf_corr_dc=np.random.multivariate_normal(mean=np.zeros(len(psrs)), cov=orfs)

        for n, psr in enumerate(psrs):
            psr.residuals_gauss += orf_corr_dc[n]*coeffs_dc*np.cos(2*np.pi*0*psr.toas)


# Kato and Soda 2016, PTAfast 2023
def gammaI00(xi, eta_reg=1e-100):
    prefactor=np.sqrt(np.pi)/2
    term_1=1 + np.cos(xi)/3
    term_2_f1=4*(1 - np.cos(xi))
    term_2_f2=np.log( np.sin(xi/2) + eta_reg )
    return prefactor*(term_1 + term_2_f1*term_2_f2)   

def gammaI10(xi, eta_reg=1e-100):
    prefactor=-np.sqrt(3*np.pi)/6
    term_1=1+np.cos(xi)
    term_2_f1=3*(1 - np.cos(xi))
    term_2_f2=1 + np.cos(xi) + 4*np.log( np.sin(xi/2) + eta_reg )
    return prefactor*(term_1 + term_2_f1*term_2_f2)

def gammaI11(xi, eta_reg=1e-100):
    prefactor=np.sqrt(6*np.pi)*np.sin(xi)/12
    term_1=1
    term_2_f1=3*(1 - np.cos(xi))
    term_2_f2=1 + 4*np.log( np.sin(xi/2) + eta_reg )/(1 + np.cos(xi) + eta_reg)
    return prefactor*(term_1 + term_2_f1*term_2_f2)

def gammaV11(xi, eta_reg=1e-100):
    prefactor=-np.sqrt(6*np.pi)/3
    factor_1=np.sin(xi)
    factor_2=1+3*((1 - np.cos(xi))/(1 + np.cos(xi) + eta_reg))*np.log( np.sin(xi/2) + eta_reg )
    return prefactor*factor_1*factor_2

# rotation matrices to bring pulsar pair a-b to computational frame 
def compute_rotation_matrices(a, b):
    # step 1: rotate `a` to align with z-axis using axis-angle rotation
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(a, z_axis)
    
    if s == 0:  # already aligned with z-axis
        R1 = np.identity(3)
    else:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R1 = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))

    # Step 2: apply R1 to b to get b'
    b_prime = np.dot(R1, b)

    # Step 3: rotate b' around z-axis so it lies in xz-plane
    x, y = b_prime[0], b_prime[1]
    r = np.hypot(x, y)
    if r == 0:
        R2 = np.identity(3)
    else:
        cos_phi = x / r
        sin_phi = y / r
        R2 = np.array([
            [cos_phi, sin_phi, 0],
            [-sin_phi, cos_phi, 0],
            [0, 0, 1]
        ])

    return R1, R2

def kin_dipole_orf_I(psrs, v_pos=np.array([0, 1, 0]), beta=1e-3, gamma=13/3):
    ni=-gamma + 2
    orfs = np.zeros((len(psrs), len(psrs)))
    for a in range(len(psrs)):
        for b in range(len(psrs)):
            # compute rotation matrices
            R1, R2 = compute_rotation_matrices(psrs[a].pos, psrs[b].pos)
            
            # transform positions and dipole direction to computational frame
            v_pos_cf = R2 @ R1 @ v_pos 
            # compute dipole angles in the computational frame
            v_phi=np.arctan2(v_pos_cf[1], v_pos_cf[0])
            v_theta = np.arccos(v_pos_cf[2])
            
            # compute angular separation and contributions
            dot_ab=np.clip(np.dot(psrs[a].pos, psrs[b].pos), -1.0, 1.0)
            xi_ab=np.arccos(dot_ab)
            hd_part = gammaI00(xi_ab)
            hd_anis_1 = gammaI10(xi_ab)*np.sqrt(4*np.pi/3)*beta*(1-ni)*np.cos(v_theta)
            hd_anis_2 = -gammaI11(xi_ab)*np.sqrt(8*np.pi/3)*beta*(1-ni)*np.sin(v_theta)*np.cos(v_phi)
            orf_ab=hd_part + hd_anis_1 + hd_anis_2
            if a == b:
                orfs[a, b] = 2*orf_ab
            else:
                orfs[a, b] = orf_ab
    return orfs

def kin_dipole_orf_V(psrs, v_pos=np.array([0, 1, 0]), beta=1e-3, gamma=13/3):
    ni=-gamma + 2
    nv=ni
    orfs = np.zeros((len(psrs), len(psrs)))
    for a in range(len(psrs)):
        for b in range(len(psrs)):
            # compute rotation matrices
            R1, R2 = compute_rotation_matrices(psrs[a].pos, psrs[b].pos)
            
            # transform positions and dipole direction to computational frame
            v_pos_cf = R2 @ R1 @ v_pos 
            # compute dipole angles in the computational frame
            v_phi=np.arctan2(v_pos_cf[1], v_pos_cf[0])
            v_theta = np.arccos(v_pos_cf[2])
            
            # compute angular separation and contributions
            dot_ab=np.clip(np.dot(psrs[a].pos, psrs[b].pos), -1.0, 1.0)
            xi_ab=np.arccos(dot_ab)
            kd_part = gammaV11(xi_ab)*np.sqrt(8*np.pi/3)*beta*(1 - nv)*np.sin(v_theta)*np.sin(v_phi)
             
            orfs[a, b] = kd_part
    return orfs


# Kinematic dipole correlation generating function
def add_kinematic_dipole(psrs, log10_A=-15, gamma=13/3, \
                         components=50, f_psd=None, \
                         fgw_min=1e-9, fgw_max=1e-7, fgw_N=10000, \
                         v_pos=np.array([0, 1, 0]), v00=1, beta=1e-3):
    '''Simulates PTA residuals with anisotropic circularly polarized GW signal
    
    Parameters:
    v00=degree of circular polarization; v00=1 is completely circularly polarized
    beta=dipole magnitude/speed in terms of v/c
    v_pos=dipole direction in Cartesian coordinates'''

    Tspan = np.amax([psr.toas.max() for psr in psrs]) - np.amin([psr.toas.min() for psr in psrs])
    npsrs=len(psrs)

    # dynamically adjust frequency components to be less than fgw_max;
    # otherwise there'd be artificial high frequency noise
    while (components / Tspan) > fgw_max:
        components -= 1

    if f_psd is None:
        f_psd = np.arange(1, components+1) / Tspan

    # for residuals_gauss
    logfgws = np.linspace(np.log(fgw_min), np.log(fgw_max), fgw_N)
    fgws=np.exp(logfgws)

    dlogfgws = (np.log(fgw_max) - np.log(fgw_min))/fgw_N
    dfgws=fgws*dlogfgws

    psd_gwb_gauss = powerlaw(fgws, log10_A=log10_A, gamma=gamma) # for residuals_gauss
    
    nn_hd=0.5/gammaI00(0) # normalize to Gamma_ab(0)=0.5
    orfs_diag = kin_dipole_orf_I(psrs, v_pos=v_pos, beta=beta, gamma=gamma)*nn_hd # diagonal terms
    orfs_cross = kin_dipole_orf_V(psrs, v_pos=v_pos, beta=beta, gamma=gamma)*nn_hd # cross diagonal terms


    for i in range(components):
        # Fourier components of the covariance matrix
        # frequency-dependent variance and cross-terms
        var_Alpha = np.sum(dfgws*TF_Alpha(fgws, i+1, i+1, Tspan)*psd_gwb_gauss)
        var_Beta  = np.sum(dfgws*TF_Beta(fgws, i+1, i+1, Tspan)*psd_gwb_gauss)
        cov_AlphaBeta = np.sum(dfgws*TF_AlphaBeta(fgws, i+1, i+1, Tspan)*v00*psd_gwb_gauss)
        cov_BetaAlpha = np.sum(dfgws*TF_BetaAlpha(fgws, i+1, i+1, Tspan)*v00*psd_gwb_gauss)

        # covariance
        Sigma = np.zeros((2*npsrs, 2*npsrs)) # correlated α–β
        Sigma_p = np.zeros((2*npsrs, 2*npsrs)) # uncorrelated α–β
        for a in range(npsrs):
            for b in range(npsrs):
                gammaI_ab = orfs_diag[a, b]
                gammaV_ab = orfs_cross[a, b]

                Sigma[a, b] = var_Alpha*gammaI_ab                  # α–α
                Sigma[npsrs + a, npsrs + b] = var_Beta*gammaI_ab  # β–β
                Sigma[a, npsrs + b] = cov_AlphaBeta*gammaV_ab     # α–β
                Sigma[npsrs + a, b] = cov_BetaAlpha*gammaV_ab     # β–α

                Sigma_p[a, b] = var_Alpha*gammaI_ab                  # α–α
                Sigma_p[npsrs + a, npsrs + b] = var_Beta*gammaI_ab  # β–β

        rng = np.random.default_rng()
        z = rng.standard_normal(2 * npsrs)
        L_Sigma = np.linalg.cholesky(Sigma)
        L_Sigma_p = np.linalg.cholesky(Sigma_p)

        coeffs_Sigma = L_Sigma @ z
        coeffs_Sigma_p = L_Sigma_p @ z

        # extract alpha and beta parts
        alpha_Sigma, beta_Sigma = coeffs_Sigma[:npsrs], coeffs_Sigma[npsrs:]
        alpha_Sigma_p, beta_Sigma_p = coeffs_Sigma_p[:npsrs], coeffs_Sigma_p[npsrs:]

        # Inject both versions
        for n, psr in enumerate(psrs):
            # Uncorrelated α–β version
            psr.residuals += alpha_Sigma_p[n]*np.sin(2*np.pi*f_psd[i]*psr.toas)
            psr.residuals += beta_Sigma_p[n]*np.cos(2*np.pi*f_psd[i]*psr.toas)

            # Correlated α–β version
            psr.residuals_gauss += alpha_Sigma[n]*np.sin(2*np.pi*f_psd[i]*psr.toas)
            psr.residuals_gauss += beta_Sigma[n]*np.cos(2*np.pi*f_psd[i]*psr.toas)


# add Roemer delay
def add_roemer_delay(psrs, planet, d_mass=0., d_Om=0., d_omega=0., d_inc=0., d_a=0., d_e=0., d_l0=0.):
    
    # check if ephem in pulsar
    for psr in psrs:
        if not hasattr(psr, 'ephem'):
            print('"ephem" not found in pulsar', psr.name)
            return

    for psr in psrs:
        psr.residuals += psr.ephem.roemer_delay(psr.toas, psr.pos, planet, d_mass, d_Om, d_omega, d_inc, d_a, d_e, d_l0)
        psr.residuals_gauss += psr.ephem.roemer_delay(psr.toas, psr.pos, planet, d_mass, d_Om, d_omega, d_inc, d_a, d_e, d_l0)

# # Noise generating function from covariance matrix
# def add_common_correlated_noise_gp(psrs, orf='hd', spectrum='powerlaw', rn_components=30, f_psd=None, h_map=None, **kwargs):

#     Tspan = np.amax([psr.toas.max() for psr in psrs]) - np.amin([psr.toas.min() for psr in psrs])
#     if f_psd is None:
#         f = np.arange(1, rn_components+1) / Tspan
#     df = np.diff(np.append(0., f))
#     if spectrum is 'custom':
#         # assert f_psd is None, '"f_psd" must not be None. The frequencies "f_psd" correspond to frequencies where the "custom_psd" is evaluated.'
#         custom_psd = kwargs['custom_psd']
#         assert len(custom_psd) == len(f), '"custom_psd" and "f_psd" must be same length. The frequencies "f_psd" correspond to frequencies where the "custom_psd" is evaluated.'
#         psd_gwb = custom_psd * df
#     elif spectrum is 'powerlaw':
#         psd_gwb = spec.powerlaw(f_psd, log10_A=kwargs['log10_A'], gamma=kwargs['gamma']) * df
#         psr.update_noisedict('common_'+orf, kwargs)
#     psd_gwb = np.repeat(psd_gwb, 2)
#     ntoas = 100
#     cov = np.zeros((len(psrs)*ntoas, len(psrs)*ntoas))
#     basis = []
#     for psr in psrs:
#         basis_psr = np.zeros((ntoas, 2*rn_components))
#         toas = np.linspace(psr.toas.min(), psr.toas.max(), ntoas)
#         for i in range(rn_components):
#             basis_psr[:, 2*i] = np.cos(2*np.pi*f[i]*toas)
#             basis_psr[:, 2*i+1] = np.sin(2*np.pi*f[i]*toas)
#         basis.append(basis_psr)
#     orf_funcs = {'hd':hd, 'monopole':monopole, 'dipole':dipole, 'curn':curn}
#     if orf in [*orf_funcs]:
#         orfs = orf_funcs[orf](psrs)
#     elif orf == 'anisotropic':
#         orfs = anisotropic(psrs, h_map)
#     for i in range(len(psrs)):
#         for j in range(len(psrs)):
#             cov_ij = np.dot(basis[i], np.dot(np.diag(orfs[i, j]*psd_gwb), basis[j].T))
#             cov[i*ntoas:(i+1)*ntoas, j*ntoas:(j+1)*ntoas] = cov_ij
#     gwb_gp = np.random.multivariate_normal(mean=np.zeros(len(psrs)*ntoas), cov=cov)
#     for k in range(len(psrs)):
#         toas = np.linspace(psrs[k].toas.min(), psrs[k].toas.max(), ntoas)
#         f = interp1d(toas, gwb_gp[k*ntoas:(k+1)*ntoas], kind='cubic')
#         psrs[k].residuals += f(psrs[k].toas)