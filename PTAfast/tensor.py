import numpy as np

# special functions and numerical integration
from scipy.special import legendre, factorial, spherical_jn
from scipy.integrate import quad

# Tensor modes

class Tensor():

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
    def djdx(self, x, l):
        v = self.v
        return np.exp(1j*x/v)*spherical_jn(l, x)/(v*(x**2))

    def Jl_T(self, l):
        v = self.v
        fD = self.fD
        r = 2*np.pi*fD*v
        if l <= 100:
            ffact = np.sqrt(factorial(l + 2)/factorial(l - 2))
            pts = []
        else:
            ffact = np.sqrt((l**4) + 2*(l**3) - (l**2) - 2*l)
            pts = np.arange(int(l*9/10), r, 1)
        im = quad(lambda x: np.imag(self.djdx(x, l)), 1e-50, r, \
                  points = pts, limit = int(1e9))[0]
        re = quad(lambda x: np.real(self.djdx(x, l)), 1e-50, r, \
                  points = pts, limit = int(1e9))[0]
        ovall = np.sqrt(2)*np.pi*(1j**l)*ffact
        return ovall*(re + 1j*im)
    
    def cl_T(self, l):
        '''returns the hd ps multipoles'''
        v = self.v
        fD = self.fD
        Jl = self.Jl_T(l)
        return Jl*np.conj(Jl)/np.sqrt(np.pi)

    def get_cls(self):
        '''generates ps multipoles up to lm'''
        cls = []
        for l in np.arange(2, self.lm + 1):
            cl = (l, self.cl_T(l))
            cls.append(cl)
        return np.array(cls).real

    # autocorrelation using RSF
    def iaa_T(self, t):
        '''returns the autocorrelation integrand in RSF'''
        v = self.v
        fD = self.fD
        vcos = 1 + v*np.cos(t)
        bare = 2*np.pi*(np.sin(t)**5)*(np.sin(np.pi*fD*vcos)**2)/ \
               (vcos**2)
        return bare/np.sqrt(4*np.pi)

    def gaa_T(self):
        '''returns autocorrelation using RSF'''
        v = self.v
        fD = self.fD
        return quad(self.iaa_T, 0, np.pi, limit = int(1e5))[0]

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
        gaa = self.gaa_T()
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
            gaa = self.gaa_T()
            tv = (gab**2) + (gaa**2)
            Tv = tv*((0.5/self.gab0_hd())**2)
            ORF_dict['TV'] = Tv
        
        return ORF_dict