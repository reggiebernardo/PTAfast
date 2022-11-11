# PTAfast

Python code for a fast calculation of the overlap reduction function in pulsar timing array (PTA) produced by the stochastic gravitational wave background for arbitrary polarizations, propagation velocities, and pulsar distances. Based on [arXiv:2208.12538](https://arxiv.org/abs/2208.12538) and [arXiv:2209.14834](https://arxiv.org/abs/2209.14834).

Please cite the above papers when using PTAFast, and let us know about any questions or comments. Happy to discuss. Thanks. - Reggie and Kin

*Minimal example* (Hellings-Downs curve): <br />
import numpy as np <br />
from PTAfast.hellingsdowns import HellingsDowns as HD <br />
lMax = 30 # maximum multipole, zeta > 6 degrees <br />
Zta = np.linspace(0, np.pi, lMax + 1) # angle space with 30 divisions <br />
hd = HD(lm = lMax) # default lm = 30 multipoles <br />
hdorf = hd.get_ORF(Zta) # add option return_tv = True for total var <br />
hdave = hdorf['ORF'] # mean <br />
hderr = np.sqrt(hdorf['CV']) # cosmic variance uncertainty

*Minimal example* (Scalar Transverse/Breathing ORF): <br />
import numpy as np <br />
from PTAfast.scalar import ScalarT as ST <br />
lMax = 30 # maximum multipole, zeta > 6 degrees <br />
Zta = np.linspace(0, np.pi, lMax + 1) # angle space with 30 divisions <br />
NEorf = ST(lm = lMax, v = 0.5, fD = 500) # v = GW speed, fD = distance <br />
orf = NEorf.get_ORF(Zta) # add option return_tv = True for total var <br />
ave = orf['ORF'] # mean <br />
err = np.sqrt(orf['CV']) # cosmic variance uncertainty
