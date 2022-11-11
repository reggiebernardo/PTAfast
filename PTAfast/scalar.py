import numpy as np

# special functions and numerical integration
from scipy.special import legendre, factorial, \
spence, spherical_jn, hyp2f1, gamma, sph_harm
from scipy.integrate import quad

# Scalar modes

class ScalarT():

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
    
    # 2nd derivative spherical Bessel, using ode
    def D2jl(self, l, x):
        jn = spherical_jn(l, x)
        jnprime = spherical_jn(l, x, derivative = True)
        return (-2*x*jnprime - (x**2 - l*(l + 1))*jn)/(x**2)
    
    # # finite fd ST ps
    def R_ST(self, x, l):
        v = self.v
        jnpp = self.D2jl(l, x)
        jn = spherical_jn(l, x)
        return -(1/np.sqrt(2))*np.exp(1j*x/v)*(jnpp + jn)/v

    def F_ST(self, l):
        v = self.v
        fD = self.fD
        r = 2*np.pi*fD*v
        re = quad(lambda x: np.real(self.R_ST(x, l)), \
                     0, r, limit = int(1e7))[0]
        im = quad(lambda x: np.imag(self.R_ST(x, l)), \
                     0, r, limit = int(1e7))[0]
        return -(1j/2)*(re + 1j*im)
    
    def cl_ST(self, l):
        '''returns the ps multipoles'''
        Proj_Fact = self.F_ST(l)
        return 32*(np.pi**2)*(Proj_Fact*np.conj(Proj_Fact))/(np.sqrt(4*np.pi))    

    def get_cls(self):
        '''generates ps multipoles up to lm'''
        cls = []
        for l in np.arange(0, self.lm + 1):
            cl = (l, self.cl_ST(l))
            cls.append(cl)
        return np.array(cls).real

    # autocorrelation using RSF
    def iaa_ST(self, t):
        v = self.v
        fD = self.fD
        vcos = 1 + v*np.cos(t)
        bare =  2*np.pi*(np.sin(t)**5)*(np.sin(np.pi*fD*vcos)**2)/(vcos**2)
        return bare/np.sqrt(4*np.pi)

    def gaa_ST(self):
        return quad(self.iaa_ST, 0, np.pi, limit = int(1e5))[0]

    # nonnormalized correlation/ORF
    def get_gab(self, zeta):
        '''returns gab for theta (array of angles)'''
        cls = self.get_cls()
        gab = [sum([(2*l + 1)*cl*legendre(l)(np.cos(phi))/(4*np.pi)
                    for l, cl in cls]) for phi in zeta]
        return np.array(gab)
    
    def get_Gab(self, zeta):
        '''returns normalized ORF'''
        gab0 = self.gab0_hd()
        Gab = self.get_gab(zeta)*0.5/gab0
        return Gab
    
    def get_Tv(self, zeta):
        '''returns variance of a single pulsar pair/total variance'''
        gab0 = self.gab0_hd()
        gaa = self.gaa_ST()
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
        '''returns normalized ORF and cosmic variance'''
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
            gaa = self.gaa_ST()
            tv = (gab**2) + (gaa**2)
            Tv = tv*((0.5/self.gab0_hd())**2)
            ORF_dict['TV'] = Tv
        
        return ORF_dict
    
class ScalarL():

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
    
    # 2nd derivative spherical Bessel, using ode
    def D2jl(self, l, x):
        jn = spherical_jn(l, x)
        jnprime = spherical_jn(l, x, derivative = True)
        return (-2*x*jnprime - (x**2 - l*(l + 1))*jn)/(x**2)
    
    # # finite fd ST ps
    def R_SL(self, x, l):
        v = self.v
        jpp = self.D2jl(l, x)
        return np.exp(1j*x/v)*jpp/v

    def F_SL(self, l):
        v = self.v
        fD = self.fD
        r = 2*np.pi*fD*v
        re = quad(lambda x: np.real(self.R_SL(x, l)), \
                  0, r, limit = int(1e7))[0]
        im = quad(lambda x: np.imag(self.R_SL(x, l)), \
                  0, r, limit = int(1e7))[0]
        return -(1j/2)*(re + 1j*im)

    def cl_SL(self, l):
        '''returns the ps multipoles'''
        Proj_Fact = self.F_SL(l)
        return 32*(np.pi**2)*(Proj_Fact*np.conj(Proj_Fact))/(np.sqrt(4*np.pi))
    
    def get_cls(self):
        '''generates ps multipoles up to lm'''
        cls = []
        for l in np.arange(0, self.lm + 1):
            cl = (l, self.cl_SL(l))
            cls.append(cl)
        return np.array(cls).real

    # autocorrelation using RSF
    def iaa_SL(self, t):
        v = self.v
        fD = self.fD
        vcos = 1 + v*np.cos(t)
        bare = 4*np.pi*np.sin(t)*(np.cos(t)**4)*(np.sin(np.pi*fD*vcos)**2)/ \
               (vcos**2)
        return bare/np.sqrt(4*np.pi)

    def gaa_SL(self):
        return quad(self.iaa_SL, 0, np.pi, limit = int(1e5))[0]

    # nonnormalized correlation/ORF
    def get_gab(self, zeta):
        '''returns gab for theta (array of angles)'''
        cls = self.get_cls()
        gab = [sum([(2*l + 1)*cl*legendre(l)(np.cos(phi))/(4*np.pi)
                    for l, cl in cls]) for phi in zeta]
        return np.array(gab)
    
    def get_Gab(self, zeta):
        '''returns normalized ORF'''
        gab0 = self.gab0_hd()
        Gab = self.get_gab(zeta)*0.5/gab0
        return Gab
    
    def get_Tv(self, zeta):
        '''returns variance of a single pulsar pair/total variance'''
        gab0 = self.gab0_hd()
        gaa = self.gaa_SL()
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
        '''returns normalized ORF and cosmic variance'''
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
            gaa = self.gaa_SL()
            tv = (gab**2) + (gaa**2)
            Tv = tv*((0.5/self.gab0_hd())**2)
            ORF_dict['TV'] = Tv
        
        return ORF_dict