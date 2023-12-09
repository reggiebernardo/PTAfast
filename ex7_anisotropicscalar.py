import numpy as np
import matplotlib.pyplot as plt
from PTAfast.scalar import *

# reproduce figure 5 (bottom-leftmost) of 2312.03383
# modify with ease to get other components

# angular space setup
lMax = 30
Zta = np.linspace(0, np.pi, lMax + 1) # 30 divisions

# v = 1.0, 0.5, 0.2 
# only dominant for scalar is l = 0, 1; very high l's are unnecessary
gab30_v1 = ScalarPhi(v = 1.0, lm = 10, fD = 1000).get_gab_anis(l = 3, m = 0, zeta = Zta)
gab30_v2 = ScalarPhi(v = 0.5, lm = 10, fD = 1000).get_gab_anis(l = 3, m = 0, zeta = Zta)
gab30_v3 = ScalarPhi(v = 0.2, lm = 10, fD = 1000).get_gab_anis(l = 3, m = 0, zeta = Zta)

# isotropic, l = m = 0
# higher components, change l, m values above

# plot intensity, others are zero
fig, ax = plt.subplots()
ax.plot(Zta*180/np.pi, gab30_v1.real, 'r-', label = r'$v = 1.0$')
ax.plot(Zta*180/np.pi, gab30_v2.real, 'b--', label = r'$v = 0.5$')
ax.plot(Zta*180/np.pi, gab30_v3.real, 'g-.', label = r'$v = 0.2$')
ax.set_ylabel(r'$\gamma_{30}(\zeta)$')
ax.set_xlabel(r'$\zeta$ [degrees]')
ax.legend(loc = 'lower right')
ax.set_xlim(min(Zta*180/np.pi), max(Zta*180/np.pi))
fig.savefig('gamma30_scalar.pdf')