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

# with cosmic variance
Dccp_cv = (3.9e-30)*np.sqrt(HD(lm = 60).get_ORF(tta)['CV']) + \
(1.5e-30)*HD(lm = 60).get_ORF(tta)['ORF']
Dccp_Total = np.sqrt(Dccp**2 + Dccp_cv**2)

# setup reference HD
lMax = 60
Zta = tta # angle separations in NG12.5

# Hellings Downs: Data + CV
def ll_hd_unc2(A2):
    # ORF of HD
    NEorf = HD(lm = lMax)
    orf = NEorf.get_ORF(Zta)
    ave = orf['ORF']
    
    # deviation
    Err2 = Dccp_Total**2
    Devi = (A2*(1e-30)*ave - ccp)/np.sqrt(Err2)
    chi2_corr = sum(Devi**2)

    return -0.5*chi2_corr

# Tensor: Data + CV
def ll_tensors_unc2_TCV(A2, v, fD = 100):
    # ORF calculation
    NEorf = Tensor(lm = lMax, v = v, fD = fD)
    orf = NEorf.get_ORF(Zta)
    ave = orf['ORF']
    
    # with cosmic variance
    Dccp_Tcv = (3.9e-30)*np.sqrt(orf['CV']) + (1.5e-30)*ave
    
    # deviation
    Err2 = Dccp**2 + Dccp_Tcv**2
    Devi = (A2*(1e-30)*ave - ccp)/np.sqrt(Err2)
    chi2_corr = sum(Devi**2)
    
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