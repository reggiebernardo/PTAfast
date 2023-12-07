from ng15correlationlikes import *
from cptacorrelationlikes import *


# Gaussian random noise, uncorrelated process
def ll_ucGRN_joint(mu, sigma):
    return ll_ucGRN_NG15(mu, sigma) + ll_ucGRN_cpta(mu, sigma)


### TENSORS

def ll_T_joint(v, fD = 500):
    return ll_T_NG15(v, fD) + ll_T_cpta(v, fD)

def ll_T_jointCV(v, fD = 500):
    return ll_T_NG15CV(v, fD) + ll_T_cptaCV(v, fD)


### vectors

def ll_V_joint(v, fD = 500):
    return ll_V_NG15(v, fD) + ll_V_cpta(v, fD)

def ll_V_jointCV(v, fD = 500):
    return ll_V_NG15CV(v, fD) + ll_V_cptaCV(v, fD)


### HD + phi

def ll_hdphi_joint(r2, v, fD = 500):
    return ll_hdphi_NG15(r2, v, fD) + ll_hdphi_cpta(r2, v, fD)

def ll_hdphi_jointCV(r2, v, fD = 500):
    return ll_hdphi_NG15CV(r2, v, fD) + ll_hdphi_cptaCV(r2, v, fD)


### T + phi

def ll_Tphi_joint(r2, v, fD = 500):
    return ll_Tphi_NG15(r2, v, fD) + ll_Tphi_cpta(r2, v, fD)

def ll_Tphi_jointCV(r2, v, fD = 500):
    return ll_Tphi_NG15CV(r2, v, fD) + ll_Tphi_cptaCV(r2, v, fD)


### HD + V

def ll_hdV_joint(r2, v, fD = 500):
    return ll_hdV_NG15(r2, v, fD) + ll_hdV_cpta(r2, v, fD)

def ll_hdV_jointCV(r2, v, fD = 500):
    return ll_hdV_NG15CV(r2, v, fD) + ll_hdV_cptaCV(r2, v, fD)