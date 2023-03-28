import numpy as np
from scipy.special import legendre

# baseline Hellings-Downs correlation

class HellingsDowns():    

    def __init__(self, lm = 30):
        self.lm = lm # l_max, largest multipole
    
    def cl_hd(self, l):
        '''returns the hd ps multipoles'''
        ovall = 8*(np.pi**(3/2))
        lfact = 1/((l + 2)*(l + 1)*l*(l - 1))
        return ovall*lfact

    def get_cls(self):
        '''generates ps multipoles up to lm'''
        cls = []
        for l in np.arange(2, self.lm + 1):
            cl = (l, self.cl_hd(l))
            cls.append(cl)
        return np.array(cls)
    
    def get_gab(self, zeta):
        '''returns gab for theta (array of angles)'''
        cls = self.get_cls()        
        gab = [sum([(2*l + 1)*cl*legendre(l)(np.cos(phi))/(4*np.pi)
                    for l, cl in cls]) for phi in zeta]
        return np.array(gab)
    
    def get_Gab(self, zeta):
        '''returns normalized HD ORF, Gab(0+) = 0.5'''
        z0 = 0.00017453292519943296 # 1e-2 deg in radians
        gab0 = self.get_gab([z0])[0]
        Gab = self.get_gab(zeta)*0.5/gab0
        return Gab
    
    def get_Tv(self, zeta):
        '''returns variance of a single pulsar pair/total variance'''
        z0 = 0.00017453292519943296 # 1e-2 deg in radians
        gab0 = self.get_gab([z0])[0]
        gaa = 2*gab0
        gab = self.get_gab(zeta)
        tv = (gab**2) + (gaa**2)
        Tv = tv*((0.5/gab0)**2)
        return Tv
        
    def get_Cv(self, zeta):
        '''returns variance of many pulsar pairs/cosmic variance'''
        z0 = 0.00017453292519943296 # 1e-2 deg in radians
        gab0 = self.get_gab([z0])[0]
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
        z0 = 0.00017453292519943296 # 1e-2 deg in radians
        gab0 = self.get_gab([z0])[0]
        gab = np.array(gab)
        cv = np.array(cv)
        Gab = gab*0.5/gab0
        Cv = cv*((0.5/gab0)**2)
        ORF_dict = {'ORF': Gab, 'CV': Cv}
        
        if return_tv == True: # if total variance is requested            
            gaa = 2*gab0
            tv = (gab**2) + (gaa**2)
            Tv = tv*((0.5/gab0)**2)
            ORF_dict['TV'] = Tv
        
        return ORF_dict