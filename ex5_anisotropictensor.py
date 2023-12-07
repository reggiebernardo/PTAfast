import numpy as np
import matplotlib.pyplot as plt
from PTAfast.tensor import Tensor

# reproduce figure 1b (top) of 2312.03383
# modify with ease to get other components

# angular space setup
lMax = 30
Zta = np.linspace(0, np.pi, lMax + 1) # 30 divisions

# v = 1.0, 0.5, 0.1
gab10_v1 = Tensor(v = 1.0, lm = 50, fD = 1000).get_gab_stokes(l = 1, m = 0, zeta = Zta)
gab10_v2 = Tensor(v = 0.5, lm = 50, fD = 1000).get_gab_stokes(l = 1, m = 0, zeta = Zta)
gab10_v3 = Tensor(v = 0.1, lm = 50, fD = 1000).get_gab_stokes(l = 1, m = 0, zeta = Zta)

# isotropic, l = m = 0
# higher components, change l, m values above
# keys = 'I', 'V', 'QpiU', 'QmiU'

# plot intensity, others are zero
fig, ax = plt.subplots()
ax.plot(Zta*180/np.pi, gab10_v1['I'].real, 'r-', label = r'$v = 1.0$')
ax.plot(Zta*180/np.pi, gab10_v2['I'].real, 'b--', label = r'$v = 0.5$')
ax.plot(Zta*180/np.pi, gab10_v3['I'].real, 'g-.', label = r'$v = 0.1$')
ax.set_ylabel(r'$\gamma_{10}^I(\zeta)$')
ax.legend(loc = 'lower right')
ax.set_xlim(min(Zta*180/np.pi), max(Zta*180/np.pi))
fig.savefig('gammaI10_tensor.pdf')