[![DOI](https://zenodo.org/badge/564638325.svg)](https://zenodo.org/badge/latestdoi/564638325) <a href="https://ascl.net/2211.001"><img src="https://img.shields.io/badge/ascl-2211.001-blue.svg?colorB=262255" alt="ascl:2211.001" /></a>
# PTAfast

Python code for ***fast*** calculation of the overlap reduction function in ***P**ulsar **T**iming **A**rray* produced by a generally anisotropic polarized stochastic gravitational wave background for arbitrary GW polarizations, GW speeds, and pulsar distances. Based on [arXiv:2208.12538](https://arxiv.org/abs/2208.12538), [arXiv:2209.14834](https://arxiv.org/abs/2209.14834), and [arXiv:2312.03383](https://arxiv.org/abs/2312.03383).

Please cite the above papers when using PTAfast, and feel free to reach out to us for any questions or comments. Thanks. - **Reggie and Kin**

Installation: `pip install PTAfast`

##### *Minimal example* (Hellings-Downs curve): <br />
> `import numpy as np` <br />
`from PTAfast.hellingsdowns import HellingsDowns as HD` <br />
`lMax = 30 # maximum multipole, zeta > 6 degrees` <br />
`Zta = np.linspace(0, np.pi, lMax + 1) # angle space with 30 divisions` <br />
`hd = HD(lm = lMax) # default lm = 30 multipoles` <br />
`hdorf = hd.get_ORF(Zta) # add option return_tv = True for total var` <br />
`hdave = hdorf['ORF'] # mean` <br />
`hderr = np.sqrt(hdorf['CV']) # cosmic variance uncertainty`

##### *Minimal example* (Scalar Transverse/Breathing ORF): <br />
> `import numpy as np` <br />
`from PTAfast.scalar import ScalarT as ST` <br />
`lMax = 30 # maximum multipole, zeta > 6 degrees` <br />
`Zta = np.linspace(0, np.pi, lMax + 1) # angle space with 30 divisions` <br />
`NEorf = ST(lm = lMax, v = 0.5, fD = 500) # v = GW speed, fD = distance` <br />
`orf = NEorf.get_ORF(Zta) # add option return_tv = True for total var` <br />
`ave = orf['ORF'] # mean` <br />
`err = np.sqrt(orf['CV']) # cosmic variance uncertainty`

##### *Minimal example* (Vector GWs in harmonic space---lower-left-Fig-1 of [2007.11009](https://arxiv.org/abs/2007.11009)): <br />
> `%matplotlib inline` <br />
`from matplotlib import pyplot as plt` <br />
`from PTAfast.vector import Vector` <br />

> `gwb_V_v1 = Vector(lm = 20, v = 1.0, fD = 1000).get_cls()` <br />
`gwb_V_v2 = Vector(lm = 20, v = 0.9, fD = 1000).get_cls()` <br />
`gwb_V_v3 = Vector(lm = 20, v = 0.8, fD = 1000).get_cls()` <br />
`gwb_V_v4 = Vector(lm = 20, v = 0.4, fD = 1000).get_cls()` <br />

> `fig, ax = plt.subplots()` <br />
`ax.plot(gwb_V_v1[:, 0], gwb_V_v1[:, 1]/gwb_V_v1[:, 1][1], 'ro', markersize = 5.0, label = r'$v = 1.0$')` <br />
`ax.plot(gwb_V_v2[:, 0], gwb_V_v2[:, 1]/gwb_V_v2[:, 1][1], 'b^', markersize = 5.0, label = r'$v = 0.9$')` <br />
`ax.plot(gwb_V_v3[:, 0], gwb_V_v3[:, 1]/gwb_V_v3[:, 1][1], 'gs', markersize = 5.0, label = r'$v = 0.8$')` <br />
`ax.plot(gwb_V_v4[:, 0], gwb_V_v4[:, 1]/gwb_V_v4[:, 1][1], 'm*', markersize = 5.0, label = r'$v = 0.4$')` <br />

> `ax.legend(loc = 'upper right', prop = {'size': 8.0})` <br />
`ax.set_yscale('log')` <br />
`ax.set_xlim(0, 20)` <br />
`ax.set_ylim(1e-7, 1e2)` <br />
`ax.grid()`

##### *Minimal example* (HD curve + anisotropy + polarization): <br />
> `import numpy as np` <br />
`from PTAfast.tensor import Tensor` <br />
`Zta = np.linspace(0, np.pi, 30 + 1) # 30 divisions` <br />
`SGWB = Tensor(v = 1.0, lm = 50, fD = 1000) # HD/LO + NLO terms` <br />
`gab00 = SGWB.get_gab_stokes(l = 0, m = 0, zeta = Zta) # HD curve` <br />
`gab10 = SGWB.get_gab_stokes(l = 1, m = 0, zeta = Zta) # NLO terms` <br />
`gab11 = SGWB.get_gab_stokes(l = 1, m = 1, zeta = Zta)` <br />
`gab20 = SGWB.get_gab_stokes(l = 2, m = 0, zeta = Zta)` <br />
`# and so on for higher anisotropic and polarized components`

##### *Other examples* (ORF for HD and nonEinsteinian GW polarizations): <br />
> See ex1_hdcurve.py, ex2_halfluminaltensor.py, etc. Run in terminal, e.g., `python ex1_hdcurve.py`.

##### *Case Study* (HD/Tensor SGWB correlation): Finite vs Infinite Distance

<table class="image" align="center" width="40%">
<tr><td><img src="https://github.com/reggiebernardo/PTAfast/blob/1a21a2e20e0f0e12e9d3102e6184a53a7d2e727e/hdlowangle.png"></td></tr>
<tr><td class="caption">SGWB spatial correlation signal for luminal nanohertz GWs given infinite (Hellings-Downs, red solid line, vertical hatches~2-sigma-cosmic variance) and finite (~30 parsecs, blue dashed line, diagonal hatches~2-sigma-cosmic variance) pulsar distances. $\zeta \in (0^\circ, 180^\circ)$ is the angle between two MSPs.</td></tr>
</table>

#### Upcoming
- Varying astrophysical pulsar distances; <br />
- RSF-PSM subdegree calculations (PTA analogy of pN-NR waveforms for compact binaries).
