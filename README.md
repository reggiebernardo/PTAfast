[![DOI](https://zenodo.org/badge/564638325.svg)](https://zenodo.org/badge/latestdoi/564638325) <a href="https://ascl.net/2211.001"><img src="https://img.shields.io/badge/ascl-2211.001-blue.svg?colorB=262255" alt="ascl:2211.001" /></a>
# PTAfast

Python code for ***fast*** calculation of the overlap reduction function in ***P**ulsar **T**iming **A**rray* produced by a generally anisotropic polarized stochastic gravitational wave background for arbitrary GW polarizations, GW speeds, and pulsar distances. Based on [arXiv:2208.12538](https://arxiv.org/abs/2208.12538), [arXiv:2209.14834](https://arxiv.org/abs/2209.14834), and [arXiv:2312.03383](https://arxiv.org/abs/2312.03383).

Please cite the above papers when using PTAfast, and feel free to reach out to us for any questions or comments. Thanks. - **Reggie and Kin**

Installation: `pip install PTAfast`

## Minimal Examples
#### Hellings & Downs curve

``` python
import numpy as np
from PTAfast.hellingsdowns import HellingsDowns as HD
lMax = 30 # maximum multipole, zeta > 6 degrees
Zta = np.linspace(0, np.pi, lMax + 1) # angle space with 30 divisions
hd = HD(lm = lMax) # default lm = 30 multipoles
hdorf = hd.get_ORF(Zta) # add option return_tv = True for total var
hdave = hdorf['ORF'] # mean
hderr = np.sqrt(hdorf['CV']) # cosmic variance uncertainty
```

#### Scalar Transverse/Breathing ORF
``` python
import numpy as np
from PTAfast.scalar import ScalarT as ST
lMax = 30 # maximum multipole, zeta > 6 degrees
Zta = np.linspace(0, np.pi, lMax + 1) # angle space with 30 divisions
NEorf = ST(lm = lMax, v = 0.5, fD = 500) # v = GW speed, fD = distance
orf = NEorf.get_ORF(Zta) # add option return_tv = True for total var
ave = orf['ORF'] # mean
err = np.sqrt(orf['CV']) # cosmic variance uncertainty
```

#### Vector GWs---lower-left-Fig-1 of [2007.11009](https://arxiv.org/abs/2007.11009)
``` python
from matplotlib import pyplot as plt
from PTAfast.vector import Vector

gwb_V_v1 = Vector(lm = 20, v = 1.0, fD = 1000).get_cls()
gwb_V_v2 = Vector(lm = 20, v = 0.9, fD = 1000).get_cls()
gwb_V_v3 = Vector(lm = 20, v = 0.8, fD = 1000).get_cls()
gwb_V_v4 = Vector(lm = 20, v = 0.4, fD = 1000).get_cls()

fig, ax = plt.subplots()
ax.plot(gwb_V_v1[:, 0], gwb_V_v1[:, 1]/gwb_V_v1[:, 1][1], 'ro', markersize = 5.0, label = r'$v = 1.0$')
ax.plot(gwb_V_v2[:, 0], gwb_V_v2[:, 1]/gwb_V_v2[:, 1][1], 'b^', markersize = 5.0, label = r'$v = 0.9$')
ax.plot(gwb_V_v3[:, 0], gwb_V_v3[:, 1]/gwb_V_v3[:, 1][1], 'gs', markersize = 5.0, label = r'$v = 0.8$')
ax.plot(gwb_V_v4[:, 0], gwb_V_v4[:, 1]/gwb_V_v4[:, 1][1], 'm*', markersize = 5.0, label = r'$v = 0.4$')

ax.legend(loc = 'upper right', prop = {'size': 8.0})
ax.set_yscale('log')
ax.set_xlim(0, 20)
ax.set_ylim(1e-7, 1e2)
ax.grid()
```

#### HD curve + anisotropy + polarization)
``` python
import numpy as np
from PTAfast.tensor import Tensor
Zta = np.linspace(0, np.pi, 30 + 1) # 30 divisions
SGWB = Tensor(v = 1.0, lm = 50, fD = 1000) # HD/LO + NLO terms
gab00 = SGWB.get_gab_stokes(l = 0, m = 0, zeta = Zta) # HD curve
gab10 = SGWB.get_gab_stokes(l = 1, m = 0, zeta = Zta) # NLO terms
gab11 = SGWB.get_gab_stokes(l = 1, m = 1, zeta = Zta)
gab20 = SGWB.get_gab_stokes(l = 2, m = 0, zeta = Zta)
# and so on for higher anisotropic and polarized components
```

#### ORF for HD and nonEinsteinian GW polarizations
See `ex1_hdcurve.py`, `ex2_halfluminaltensor.py`, etc. In terminal, e.g., `python ex1_hdcurve.py`.

## Case study: Pulsar term modulations at finite distance

<p align="center">
  <img src="https://github.com/reggiebernardo/PTAfast/blob/1a21a2e20e0f0e12e9d3102e6184a53a7d2e727e/hdlowangle.png" width="50%">
</p>

<p align="center" style="width: 50%">
  <i>Inter-pulsar spatial correlation for nanohertz GWs with pulsars at infinity (Hellings-Downs, red solid, hatches~2-sigma-cosmic variance) and finite distance (~30 parsecs, blue dashed, hatches~2-sigma-cosmic variance). More at https://arxiv.org/abs/2209.14834.</i>
</p>

## Upcoming
- Varying astrophysical pulsar distances <br />
- Angular control for more efficient calculation <br />
- RSF-PSM subdegree calculations (PTA analogy of pN-NR waveforms for compact binaries)
