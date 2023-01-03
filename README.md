[![DOI](https://zenodo.org/badge/564638325.svg)](https://zenodo.org/badge/latestdoi/564638325)
# PTAfast

Python code for ***fast*** calculation of the overlap reduction function in ***P**ulsar **T**iming **A**rray* produced by the stochastic gravitational wave background for arbitrary polarizations, propagation velocities, and pulsar distances. Based on [arXiv:2208.12538](https://arxiv.org/abs/2208.12538) and [arXiv:2209.14834](https://arxiv.org/abs/2209.14834).

Please cite the above papers when using PTAfast, and let us know about any questions or comments. Always happy to discuss. Thanks. - **Reggie and Kin**

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

##### *Other examples* (ORF for HD and nonEinsteinian GW polarizations): <br />
> See ex1_hdcurve.py, ex2_halfluminaltensor.py, etc. Run in terminal, e.g., `python ex1_hdcurve.py`.

##### *Case Study* (HD/Tensor SGWB correlation): Finite vs Infinite Distance

<table class="image" align="center" width="50%">
<tr><td><img src="https://github.com/reggiebernardo/PTAfast/blob/1a21a2e20e0f0e12e9d3102e6184a53a7d2e727e/hdlowangle.png"></td></tr>
<tr><td class="caption">The PTA spatial correlation signal produced by light speed nanohertz GWs given infinite (Hellings-Downs/HD, red solid line, vertical hatches for the variance) and finite (30 parsecs, blue dashed line, diagonal hatches for the variance) pulsar distances. $\zeta \in (0^\circ, 180^\circ)$ is the angle between two MSPs. The lines correspond to the mean of the correlation while the colored bands show the two-sigma cosmic variance uncertainty.</td></tr>
</table>
