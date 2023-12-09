import numpy as np
import matplotlib.pyplot as plt
from PTAfast.vector import Vector

# reproduce figure 5 (bottom-rightmost) of 2312.03383
# modify with ease to get other components

# angular space setup
lMax = 30
Zta = np.linspace(0, np.pi, lMax + 1) # 30 divisions

# note: v = 1 is ruled out for vectors
# v = 0.9, 0.5, 0.1
gab22_v1 = Vector(v = 0.9, lm = 50, fD = 1000).get_gab_stokes(l = 2, m = 2, zeta = Zta)
gab22_v2 = Vector(v = 0.5, lm = 50, fD = 1000).get_gab_stokes(l = 2, m = 2, zeta = Zta)
gab22_v3 = Vector(v = 0.1, lm = 50, fD = 1000).get_gab_stokes(l = 2, m = 2, zeta = Zta)

# isotropic, l = m = 0
# higher components, change l, m values above
# keys = 'I', 'V', 'QpiU', 'QmiU'

# plot intensity, others are zero
fig, ax = plt.subplots()
ax.plot(Zta*180/np.pi, gab22_v1['QmiU'].real, 'r-', label = r'$v = 0.9$')
ax.plot(Zta*180/np.pi, gab22_v2['QmiU'].real, 'b--', label = r'$v = 0.5$')
ax.plot(Zta*180/np.pi, gab22_v3['QmiU'].real, 'g-.', label = r'$v = 0.1$')
ax.set_ylabel(r'$\gamma_{22}^{Q-iU}(\zeta)$')
ax.set_xlabel(r'$\zeta$ [degrees]')
ax.legend(loc = 'lower left')
ax.set_xlim(min(Zta*180/np.pi), max(Zta*180/np.pi))
fig.savefig('gammaQmiU22_vector.pdf')