import numpy as np


# time kernels

def C_T(f, ta, tb):
    w=2*np.pi*f
    return 1 - np.cos(w*ta) - np.cos(w*tb) + np.cos(w*(tb-ta))

def S_T(f, ta, tb):
    w=2*np.pi*f
    return np.sin(w*ta) - np.sin(w*tb) + np.sin(w*(tb-ta))


# transfer functions

def TF_Alpha(f, k1, k2, T):
    '''PTA transfer function---low pass filter'''
    # sin=lambda x: np.sin(x)
    # pi=np.pi
    return 4*k1*k2*np.sin(np.pi*T*f)**2/(np.pi**2*(-T**2*f**2 + k1**2)*(-T**2*f**2 + k2**2))

def TF_Beta(f, k1, k2, T):
    '''PTA transfer function---high pass filter'''
    # sin=lambda x: np.sin(x)
    # pi=np.pi
    return 4*T**2*f**2*np.sin(np.pi*T*f)**2/(np.pi**2*(T**2*f**2 - k1**2)*(T**2*f**2 - k2**2))

def TF_AlphaBeta(f, k1, k2, T):
    '''PTA transfer function---circular polarization filter'''
    return -4*k1*f*T*(np.sin(np.pi*f*T)**2)/((np.pi**2)*((k1**2)-((f*T)**2))*((k2**2)-((f*T)**2)))

def TF_BetaAlpha(f, k1, k2, T):
    '''PTA transfer function---circular polarization filter'''
    return 4*k2*f*T*(np.sin(np.pi*f*T)**2)/((np.pi**2)*((k1**2)-((f*T)**2))*((k2**2)-((f*T)**2)))


# transfer functions: static and cross-static bins

def Ca0a0(f, T):
    fT=f*T
    FP=fT%1 # fractional part
    IP=fT-FP # integer part
    t1=4*np.sin(np.pi*f*T)*np.sin(np.pi*FP)*np.cos(np.pi*IP)
    t2=4*np.pi*f*T*( np.pi*f*T - np.sin(2*np.pi*f*T) )
    return (t1 + t2)/( (np.pi*f*T)**2 )

def Ca0ak(f, k, T):
    return -4*k*( np.sin(np.pi*f*T)**2 )/( np.pi*( (k**2)-((f*T)**2) ) )

def Ca0bk(f, k, T):
    return 2*( np.pi*f*T*np.sin(2*np.pi*f*T) + np.cos(2*np.pi*f*T) - 1 )/( (np.pi**2)*( (k**2)-((f*T)**2) ) )

