import numpy as np
from scipy.special import legendre, spherical_jn

from PTAfast.hellingsdowns import HellingsDowns as HD
from PTAfast.tensor import Tensor
from PTAfast.vector import Vector
from PTAfast.scalar import ScalarL as SL
from PTAfast.scalar import ScalarT as ST


# import binned correlations data
ng15 = np.array([[ 1.33750e+01,  4.35000e-01,  2.70000e-01], \
                 [ 2.37500e+01,  2.20000e-01,  2.27000e-01], \
                 [ 3.28125e+01,  3.85000e-01,  1.93000e-01], \
                 [ 4.12500e+01,  2.57000e-01,  1.77000e-01], \
                 [ 4.72500e+01,  4.10000e-01,  2.15000e-01], \
                 [ 5.56250e+01,  1.50000e-01,  1.50000e-01], \
                 [ 6.57500e+01, -5.00000e-03,  2.35000e-01], \
                 [ 7.66250e+01,  1.20000e-02,  2.41000e-01], \
                 [ 8.70000e+01, -7.60000e-02,  2.42000e-01], \
                 [ 9.78750e+01, -8.50000e-02,  2.60000e-01], \
                 [ 1.07500e+02, -8.10000e-02,  2.47000e-01], \
                 [ 1.18750e+02,  3.50000e-02,  1.40000e-01], \
                 [ 1.32500e+02,  2.37000e-01,  2.18000e-01], \
                 [ 1.45875e+02,  8.00000e-02,  2.80000e-01], \
                 [ 1.62375e+02,  4.70000e-01,  3.17000e-01]])

# cross correlations
zab_Deg = ng15[:, 0] # degrees
zab = zab_Deg*np.pi/180
Gab_NG15 = ng15[:, 1]
DGab_NG15 = ng15[:, 2]


# normalize hd, later for the scalar field
hd1k = HD(lm = 1000)
cls_hd1k = hd1k.get_cls()
gab0_hd1k = sum([(2*l + 1)*cl/(4*np.pi) for l, cl in cls_hd1k])


# setup max Cl's
lMax = 60
hd = HD(lm = lMax)


# tensors
def ll_T_NG15(v, fD = 500):
    # ORF calculation
    orf = Tensor(lm = lMax, v = v, fD = fD).get_ORF(zab)
    ave = orf['ORF']
    
    # deviation
    Err = DGab_NG15
    Dev = (ave - Gab_NG15)/Err
    chi2 = sum(Dev**2)
    
    return -0.5*chi2

def ll_T_NG15CV(v, fD = 500):
    # ORF calculation
    orf = Tensor(lm = lMax, v = v, fD = fD).get_ORF(zab)
    ave = orf['ORF']
    CV = orf['CV'] # cosmic variance
    
    # deviation
    Err = np.sqrt(DGab_NG15**2 + CV)
    Dev = (ave - Gab_NG15)/Err
    chi2 = sum(Dev**2)
    
    return -0.5*chi2


# vectors
def ll_V_NG15(v, fD = 500):
    # ORF calculation
    orf = Vector(lm = lMax, v = v, fD = fD).get_ORF(zab)
    ave = orf['ORF']
    
    # deviation
    Err = DGab_NG15
    Dev = (ave - Gab_NG15)/Err
    chi2 = sum(Dev**2)
    
    return -0.5*chi2

def ll_V_NG15CV(v, fD = 500):
    # ORF calculation
    orf = Vector(lm = lMax, v = v, fD = fD).get_ORF(zab)
    ave = orf['ORF']
    CV = orf['CV'] # cosmic variance
    
    # deviation
    Err = np.sqrt(DGab_NG15**2 + CV)
    Dev = (ave - Gab_NG15)/Err
    chi2 = sum(Dev**2)
    
    return -0.5*chi2


# scalar field relations, Cl's, ORF
def FF_phi(v, l, fD = 500):
    '''Galileon projection factor for GW speed v with fixed distance fD'''
    lM = 10 # shouldn't need too many multipoles for the scalar field
    ff_st = ST(lm = lM, v = v, fD = fD).F_ST(l)
    ff_sl = SL(lm = lM, v = v, fD = fD).F_SL(l)
    mix = (1 - v**2)/np.sqrt(2)
    return ff_st + mix*ff_sl

def cls_phi(v, fD = 500):
    lM = 10 # shouldn't need too many multipoles for the scalar field
    Cls = []
    for l in np.arange(0, lM + 1, 1):
        ff_phi = FF_phi(v = v, l = l, fD = fD)
        cl = 32*(np.pi**2)*(ff_phi*np.conj(ff_phi))/(np.sqrt(4*np.pi))
        Cls.append([l, cl.real])
    return np.array(Cls)

def ORF_phi(zeta, v, fD = 500, get_cv = False):
    Cls = cls_phi(v, fD)
    gab = [sum([(2*l + 1)*cl*legendre(l)(np.cos(phi))/(4*np.pi)
                for l, cl in Cls]) for phi in zeta]
    ORF_dict = {'ORF': np.array(gab)*0.5/gab0_hd1k}
    if get_cv == True:
        cv = [sum([(2*l + 1)*((cl*legendre(l)(np.cos(phi)))**2)/(8*(np.pi**2)) \
                   for l, cl in Cls]) for phi in zeta]
        CV = np.array(cv)*((0.5/gab0_hd1k)**2)
        ORF_dict['CV'] = CV
    return ORF_dict


# hd + phi, e.g., Horndeski + GB
def ll_hdphi_NG15(r2, v, fD = 500):
    # ORF calculation
    orf_phi = ORF_phi(zab, v = v, fD = fD)
    orf_hd = hd.get_ORF(zab)
    ave = orf_hd['ORF'] + r2*orf_phi['ORF']
    
    # deviation
    Err = DGab_NG15
    Dev = (ave - Gab_NG15)/Err
    chi2 = sum(Dev**2)
    
    return -0.5*chi2

def ll_hdphi_NG15CV(r2, v, fD = 500):
    # ORF calculation
    orf_phi = ORF_phi(zab, v = v, fD = fD, get_cv = True)
    orf_hd = hd.get_ORF(zab)
    ave = orf_hd['ORF'] + r2*orf_phi['ORF']
    CV = orf_hd['CV'] + (r2**2)*orf_phi['CV'] # cosmic variance
    
    # deviation
    Err = np.sqrt(DGab_NG15**2 + CV)
    Dev = (ave - Gab_NG15)/Err
    chi2 = sum(Dev**2)
    
    return -0.5*chi2


# hd + V
def ll_hdV_NG15(r2, v, fD = 500):
    # ORF calculation    
    orf_V = Vector(lm = lMax, v = v, fD = fD).get_ORF(zab)
    orf_hd = hd.get_ORF(zab)
    ave = orf_hd['ORF'] + r2*orf_V['ORF']
    
    # deviation
    Err = DGab_NG15
    Dev = (ave - Gab_NG15)/Err
    chi2 = sum(Dev**2)
    
    return -0.5*chi2

def ll_hdV_NG15CV(r2, v, fD = 500):
    # ORF calculation
    orf_V = Vector(lm = lMax, v = v, fD = fD).get_ORF(zab)
    orf_hd = hd.get_ORF(zab)
    ave = orf_hd['ORF'] + r2*orf_V['ORF']
    CV = orf_hd['CV'] + (r2**2)*orf_V['CV'] # cosmic variance
    
    # deviation
    Err = np.sqrt(DGab_NG15**2 + CV)
    Dev = (ave - Gab_NG15)/Err
    chi2 = sum(Dev**2)
    
    return -0.5*chi2


# T + S, e.g., massive gravity
def ll_Tphi_NG15(r2, v, fD = 500):
    # ORF calculation
    orf_T = Tensor(lm = lMax, v = v, fD = fD).get_ORF(zab)
    orf_phi = ORF_phi(zab, v = v, fD = fD)
    ave = orf_T['ORF'] + r2*orf_phi['ORF']
    
    # deviation
    Err = DGab_NG15
    Dev = (ave - Gab_NG15)/Err
    chi2 = sum(Dev**2)
    
    return -0.5*chi2

def ll_Tphi_NG15CV(r2, v, fD = 500):
    # ORF calculation
    orf_phi = ORF_phi(zab, v = v, fD = fD, get_cv = True)
    orf_T = Tensor(lm = lMax, v = v, fD = fD).get_ORF(zab)
    ave = orf_T['ORF'] + r2*orf_phi['ORF']
    CV = orf_T['CV'] + (r2**2)*orf_phi['CV'] # cosmic variance
    
    # deviation
    Err = np.sqrt(DGab_NG15**2 + CV)
    Dev = (ave - Gab_NG15)/Err
    chi2 = sum(Dev**2)
    
    return -0.5*chi2


# derived variables
def m_g(v):
    '''graviton mass in 10^{-22} eV'''
    h_evy = 1.310542 # \times 10^{-22} eV year
    f_ref = 1 # yr^{-1}, PTA reference frequency
    mg_2 = ((h_evy*f_ref)**2)*(1 - v**2)
    return np.sqrt(mg_2)
    

# Gaussian random noise, uncorrelated process
def ll_ucGRN_NG15(mu, sigma):
    Gab = np.random.normal(loc = mu, scale = sigma, size = len(zab))
    chi2_corr = sum(((Gab - Gab_NG15)/DGab_NG15)**2)
    
    return -0.5*chi2_corr