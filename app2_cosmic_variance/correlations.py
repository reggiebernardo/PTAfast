import numpy as np
from numpy.random import multivariate_normal as MN
from PTAfast.tensor import Tensor
from PTAfast.hellingsdowns import HellingsDowns as HD

# import binned correlations data
data_loc = 'orf_n12p5.txt'
ng12 = np.loadtxt(data_loc)

# cross correlations
tta = ng12[0] # radians 
Dtta = ng12[1] # radians
ccp = ng12[2]
Dccp = ng12[3]

# setup reference HD
lMax = 60
Zta = tta # angle separations in NG12.5

# Hellings Downs likelihood with various uncertainties
def ll_hd_unc0(A2):    
    # ORF of HD, no uncertainty
    NEorf = HD(lm = lMax)
    orf = NEorf.get_ORF(Zta)
    ave = orf['ORF']
    
    # deviation
    Err2 = Dccp**2
    Devi = (A2*(1e-30)*ave - ccp)/np.sqrt(Err2)
    chi2_corr = sum(Devi**2)

    return -0.5*chi2_corr

def ll_hd_unc2(A2):
    # ORF of HD, HD + CV
    NEorf = HD(lm = lMax)
    orf = NEorf.get_ORF(Zta)
    ave = orf['ORF']
    err = np.sqrt(orf['CV'])
    drw = MN(ave, np.diag(err**2))
    
    # deviation
    Err2 = Dccp**2
    Devi = (A2*(1e-30)*drw - ccp)/np.sqrt(Err2)
    chi2_corr = sum(Devi**2)

    return -0.5*chi2_corr


# Tensor likelihood given various uncertainty models
def ll_tensors_unc0(A2, v, fD = 100):
    # ORF calculation, no uncertainty
    NEorf = Tensor(lm = lMax, v = v, fD = fD)
    orf = NEorf.get_ORF(Zta)
    ave = orf['ORF']
    
    # deviation
    Err2 = Dccp**2
    Devi = (A2*(1e-30)*ave - ccp)/np.sqrt(Err2)
    chi2_corr = sum(Devi**2)
    
    return -0.5*chi2_corr

def ll_tensors_unc2(A2, v, fD = 100):
    # ORF calculation, T + CV
    NEorf = Tensor(lm = lMax, v = v, fD = fD)
    orf = NEorf.get_ORF(Zta)
    ave = orf['ORF']
    err = np.sqrt(orf['CV'])
    drw = MN(ave, np.diag(err**2))
    
    # deviation
    Err2 = Dccp**2
    Devi = (A2*(1e-30)*drw - ccp)/np.sqrt(Err2)
    chi2_corr = sum(Devi**2)
    
    return -0.5*chi2_corr

# monopole
def ll_mon(A2):    
    # syst mon
    Gab = 0.5
    chi2_corr = sum(((A2*(1e-30)*Gab - ccp)/Dccp)**2)

    return -0.5*chi2_corr

# uncorrelated (Gaussian random) noise
def ll_GRN(A2, sigma):
    Gab = np.random.normal(loc = 0.5, scale = sigma, size = len(tta))
    chi2_corr = sum(((A2*(1e-30)*Gab - ccp)/Dccp)**2)
    
    return -0.5*chi2_corr

# derived variables
def m_g(v):
    '''velocity to graviton mass in 10^{-22} eV'''
    h_evy = 1.310542e-22 # electron volts year
    f_pta = 1 # per year
    mg_2 = ((h_evy*f_pta)**2)*(1 - v**2)
    return np.sqrt(mg_2)*(1e22)

def A(A2):
    '''GWB amplitude (\times 1e-15)'''
    return np.sqrt(A2)