import numpy as np
from PTAfast.hellingsdowns import HellingsDowns as HD

# import mock data
data_loc = 'mock.txt'
data = np.loadtxt(data_loc)

# mock cross correlations
tta = data[0]
ccp = data[1]
Dccp = data[2]

# with cosmic variance
Dccp_cv = (1e-30)*np.sqrt(HD(lm = 60).get_ORF(tta)['CV']) + \
0.1*(1e-30)*HD(lm = 60).get_ORF(tta)['ORF']
Dccp_Total = np.sqrt(Dccp**2 + Dccp_cv**2)

# setup reference HD
lMax = 60
Zta = tta # angle separations in mock

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

# derived variables
def A(A2):
    '''GWB amplitude (\times 1e-15)'''
    return np.sqrt(A2)