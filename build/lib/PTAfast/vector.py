import numpy as np

# special functions and numerical integration
from scipy.special import legendre, factorial, spherical_jn, gamma
from scipy.integrate import quad

# Tensor modes

class Vector():

    def __init__(self, lm = 30, v = 1, fD = 100):
        self.lm = lm # l_max, largest multipole
        self.v = v # propagation speed
        self.fD = fD # pulsar distance

    # HD values for reference
    def cl_hd(self, l):
        '''returns the hd ps multipoles'''
        ovall = 8*(np.pi**(3/2))
        lfact = 1/((l + 2)*(l + 1)*l*(l - 1))
        return ovall*lfact
    
    def gab0_hd(self):
        gab0 = sum([(2*l + 1)*self.cl_hd(l)/(4*np.pi) \
                    for l in np.arange(2, self.lm + 1)])
        return gab0
    
    # # finite fd tensor ps
    def IV(self, x, l):
        v = self.v
        return np.exp(1j*x/v)*spherical_jn(l, x)/(x*v)

    def JV(self, l):
        v = self.v
        fD = self.fD
        prefact = 2*np.sqrt(2)*np.pi*(1j**l)*np.sqrt(l*(l + 1))
        r = 2*np.pi*fD*v
        re = quad(lambda x: np.real(self.IV(x, l)), \
                  0, r, limit = int(1e7))[0]
        im = quad(lambda x: np.imag(self.IV(x, l)), \
                  0, r, limit = int(1e7))[0]
        y = 2*np.pi*fD
        bt1 = np.exp(1j*y)*spherical_jn(l, y*v)/(y*(v**2))
        tot = (-1j/v)*(re + 1j*im) + bt1
        if l == 1:
            bt2 = -np.sqrt(np.pi)/(v*(2**(l + 1))*gamma(l + (3/2)))
            tot += bt2
        return prefact*tot

    def cl_V(self, l):
        v = self.v
        fD = self.fD
        Jl = self.JV(l)
        return Jl*np.conj(Jl)/np.sqrt(np.pi)

    def get_cls(self):
        '''generates ps multipoles up to l <= lm'''
        cls = []
        for l in np.arange(1, self.lm + 1):
            cl = (l, self.cl_V(l))
            cls.append(cl)
        return np.array(cls).real

    # autocorrelation using RSF
    def iaa_v(self, t):
        '''returns the autocorrelation integrand in RSF'''
        v = self.v
        fD = self.fD
        vcos = 1 + v*np.cos(t)
        bare = 8*np.pi*(np.sin(t)**3)*(np.cos(t)**2)* \
               (np.sin(np.pi*fD*vcos)**2)/(vcos**2)
        return bare/np.sqrt(4*np.pi)

    def gaa_V(self):
        '''returns autocorrelation using RSF'''
        return quad(self.iaa_v, 0, np.pi, limit = int(1e5))[0]
    
    # nonnormalized correlation/ORF
    def get_gab(self, zeta):
        '''returns gab for theta (array of angles)'''
        cls = self.get_cls()
        gab = [sum([(2*l + 1)*cl*legendre(l)(np.cos(phi))/(4*np.pi)
                    for l, cl in cls]) for phi in zeta]
        return np.array(gab)
    
    def get_Gab(self, zeta):
        '''returns normalized HD ORF, Gab(0+) = 0.5'''
        gab0 = self.gab0_hd()
        Gab = self.get_gab(zeta)*0.5/gab0
        return Gab
    
    def get_Tv(self, zeta):
        '''returns variance of a single pulsar pair/total variance'''
        gab0 = self.gab0_hd()
        gaa = self.gaa_V()
        gab = self.get_gab(zeta)
        tv = (gab**2) + (gaa**2)
        Tv = tv*((0.5/gab0)**2)
        return Tv
        
    def get_Cv(self, zeta):
        '''returns variance of many pulsar pairs/cosmic variance'''
        gab0 = self.gab0_hd()
        cls = self.get_cls()
        cv = [sum([(2*l + 1)*((cl*legendre(l)(np.cos(phi)))**2)/(8*(np.pi**2))
                   for l, cl in cls]) for phi in zeta]
        Cv = np.array(cv)*((0.5/gab0)**2)
        return Cv
    
    # master function for ORF and cosmic variance
    def get_ORF(self, zeta, return_tv = False):
        cls = self.get_cls()
        
        # terms in mean and cosmic variance
        gab = [] # phi dependence
        cv = []
        for phi in zeta:
            gab_i = [] # terms per zeta
            cv_i = []
            for l, cl in cls:
                gab_i.append((2*l + 1)*cl*legendre(l)(np.cos(phi))/(4*np.pi))
                cv_i.append((2*l + 1)*((cl*legendre(l)(np.cos(phi)))**2)/(8*(np.pi**2)))
            gab.append(sum(gab_i))
            cv.append(sum(cv_i))            
        
        # normalized ORF
        gab = np.array(gab)
        cv = np.array(cv)
        Gab = gab*0.5/self.gab0_hd()
        Cv = cv*((0.5/self.gab0_hd())**2)
        ORF_dict = {'ORF': Gab, 'CV': Cv}
        
        if return_tv == True: # if total variance is requested
            gaa = self.gaa_V()
            tv = (gab**2) + (gaa**2)
            Tv = tv*((0.5/self.gab0_hd())**2)
            ORF_dict['TV'] = Tv
        
        return ORF_dict