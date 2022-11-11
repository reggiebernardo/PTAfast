import numpy as np
import matplotlib.pyplot as plt
from PTAfast.hellingsdowns import HellingsDowns as HD

# minimal example for HD ORF
lMax = 30
Zta = np.linspace(0, np.pi, lMax + 1) # angle space with 30 divisions
hd = HD(lm = lMax) # default lm = 30 multipoles

# make plot
hdorf = hd.get_ORF(Zta) # add option return_tv = True for total var
hdave = hdorf['ORF']
hderr = np.sqrt(hdorf['CV'])

fig = plt.figure()
plt.plot(Zta*180/np.pi, hdave, 'r-', label = 'ORF')
plt.fill_between(Zta*180/np.pi, hdave - 2*hderr, hdave + 2*hderr, \
                 alpha = 0.2, color = 'red', facecolor = 'red', \
                 edgecolor = 'red', hatch = '|', label = r'$2*\sigma_{\rm CV}$')
plt.xlim(180/lMax, 180)
plt.xlabel(r'$\zeta$ [degrees]')
plt.ylabel(r'$\Gamma_{ab}$')
plt.legend(loc = 'upper right', prop = {'size': 10})
fig.set_rasterized(True)
fig.savefig('ex1_hdcurve.pdf', dpi = 300)