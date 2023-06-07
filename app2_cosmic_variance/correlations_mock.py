import numpy as np
from numpy.random import multivariate_normal as MN
from PTAfast.hellingsdowns import HellingsDowns as HD

# import mock data
data_loc = 'mock.txt'
data = np.loadtxt(data_loc)

# mock cross correlations
tta = data[0]
ccp = data[1]
Dccp = data[2]

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
    # ORF of HD, Data + CV
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

# monopole
def ll_mon(A2):    
    # syst mon
    Gab = 0.5
    chi2_corr = sum(((A2*(1e-30)*Gab - ccp)/Dccp)**2)

    return -0.5*chi2_corr

# uncorrelated Gaussian random noise
def ll_GRN(A2, sigma):
    Gab = np.random.normal(loc = 0.5, scale = sigma, size = len(tta))
    chi2_corr = sum(((A2*(1e-30)*Gab - ccp)/Dccp)**2)
    
    return -0.5*chi2_corr

# derived variable
def A(A2):
    '''GWB amplitude (\times 1e-15)'''
    return np.sqrt(A2)