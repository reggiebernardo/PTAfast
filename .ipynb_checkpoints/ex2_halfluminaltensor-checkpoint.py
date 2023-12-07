import numpy as np
import matplotlib.pyplot as plt
from PTAfast.hellingsdowns import HellingsDowns as HD
from PTAfast.tensor import Tensor

# setup reference HD
lMax = 30
Zta = np.linspace(0, np.pi, lMax + 1) # angle space with 30 divisions
hd = HD(lm = lMax) # default lm = 30 multipoles

# make plot
hdorf = hd.get_ORF(Zta) # add option return_tv = True for total var
hdave = hdorf['ORF']
hderr = np.sqrt(hdorf['CV'])

fig = plt.figure()
plt.plot(Zta*180/np.pi, hdave, 'r-', label = 'HD')
plt.fill_between(Zta*180/np.pi, hdave - 2*hderr, hdave + 2*hderr, \
                 alpha = 0.2, color = 'red', facecolor = 'red', \
                 edgecolor = 'red', hatch = '|', label = r'HD $2\sigma_{\rm CV}$')

# calculate ORF of half luminal tensor
NEorf = Tensor(lm = lMax, v = 0.5, fD = 200) # v = GW speed, fD = distance

# make plot
orf = NEorf.get_ORF(Zta) # add option return_tv = True for total var
ave = orf['ORF']
err = np.sqrt(orf['CV'])

plt.plot(Zta*180/np.pi, ave, 'b--', label = 'ORF')
plt.fill_between(Zta*180/np.pi, ave - 2*err, ave + 2*err, \
                 alpha = 0.2, color = 'blue', facecolor = 'blue', \
                 edgecolor = 'blue', hatch = '/', label = r'ORF $2\sigma_{\rm CV}$')
plt.xlim(180/lMax, 180)
plt.xlabel(r'$\zeta$ [degrees]')
plt.ylabel(r'$\Gamma_{ab}$')
plt.legend(loc = 'upper right', prop = {'size': 10})
plt.title(r'Tensor, $v$ = ' + str(NEorf.v))
fig.set_rasterized(True)
fig.savefig('ex2_halfluminaltensor.pdf', dpi = 300)