import numpy as np
from PTAfast.tensor import Tensor
from PTAfast.vector import Vector
from PTAfast.hellingsdowns import HellingsDowns as HD

# import data
data_loc = 'https://raw.githubusercontent.com/reggiebernardo/galileon_pta/main/orf_n12p5.txt'
n12p5 = np.loadtxt(data_loc)

tta_rad = n12p5[0]
tta = tta_rad*180/np.pi
Dtta = n12p5[1]*180/np.pi
gma = n12p5[2]
Dgma = n12p5[3]

# common spectrum process
CSP = {'ave': 1.92e-15, 'err': max((2.67 - 1.92)*1e-15, (1.92 - 1.37)*1e-15)/2}
A2Gma_ng125 = {'ave': CSP['ave']**2, 'err': 2*CSP['ave']*CSP['err'] + CSP['err']**2}

# setup reference HD
lMax = 60
Zta = tta_rad # angle separations in NG12.5

def chi2_tensors(A2, v, fD = 100):    
    # calculate ORF of half luminal tensor
    NEorf = Tensor(lm = lMax, v = v, fD = fD) # v = GW speed, fD = distance

    orf = NEorf.get_ORF(Zta) # add option return_tv = True for total var
    ave = orf['ORF']
    err = np.sqrt(orf['CV'])
    
    Err2 = (A2*err)**2 + Dgma**2
    Devi = (A2*ave - gma)/np.sqrt(Err2)
    chi2_corr = sum(Devi**2)
    
    Gaa = NEorf.gaa_T()/Tensor(lm = 10, v = 1, fD = fD).gaa_T()
    model_csp = A2*Gaa
    chi2_csp = ((model_csp - A2Gma_ng125['ave'])/A2Gma_ng125['err'])**2

    return chi2_corr + chi2_csp

def loglike_tensors(A2, v, fD = 100):
    return -0.5*chi2_tensors(A2, v, fD)

# vectors

def chi2_vectors(A2, v, fD = 100):    
    # calculate ORF of vector
    NEorf = Vector(lm = lMax, v = v, fD = fD) # v = GW speed, fD = distance

    orf = NEorf.get_ORF(Zta) # add option return_tv = True for total var
    ave = orf['ORF']
    err = np.sqrt(orf['CV'])
    
    Err2 = (A2*err)**2 + Dgma**2
    Devi = (A2*ave - gma)/np.sqrt(Err2)
    chi2_corr = sum(Devi**2)
    
    Gaa = NEorf.gaa_V()/Vector(lm = 10, v = 1, fD = fD).gaa_V()
    model_csp = A2*Gaa
    chi2_csp = ((model_csp - A2Gma_ng125['ave'])/A2Gma_ng125['err'])**2

    return chi2_corr + chi2_csp

def loglike_vectors(A2, v, fD = 100):
    return -0.5*chi2_vectors(A2, v, fD)

# Hellings Downs curve

def chi2_HD(A2):    
    # calculate ORF of HD
    NEorf = HD(lm = lMax)

    orf = NEorf.get_ORF(Zta) # add option return_tv = True for total var
    ave = orf['ORF']
    err = np.sqrt(orf['CV'])
    
    Err2 = (A2*err)**2 + Dgma**2
    Devi = (A2*ave - gma)/np.sqrt(Err2)
    chi2_corr = sum(Devi**2)
    
    Gaa = 1
    model_csp = A2*Gaa
    chi2_csp = ((model_csp - A2Gma_ng125['ave'])/A2Gma_ng125['err'])**2

    return chi2_corr + chi2_csp

def loglike_HD(A2):
    return -0.5*chi2_HD(A2)

# GW monopole

def chi2_mon(A2):    
    # calculate ORF of HD
    Gab = 0.5
    chi2_corr = sum(((A2*Gab - gma)/Dgma)**2)
    
    Gaa = 1
    model_csp = A2*Gaa
    chi2_csp = ((model_csp - A2Gma_ng125['ave'])/A2Gma_ng125['err'])**2

    return chi2_corr + chi2_csp

def loglike_mon(A2):
    return -0.5*chi2_mon(A2)

# derived variables

def mg(v):
    '''velocity to graviton mass in 10^{-22} eV'''
    h_evy = 1.310542e-22 # electron volts year
    f_pta = 1 # per year
    mg_2 = ((h_evy*f_pta)**2)*(1 - v**2)
    return np.sqrt(mg_2)*(1e22)

def A(A2):
    '''GWB amplitude'''
    return np.sqrt(A2)