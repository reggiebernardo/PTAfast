import numpy as np
import matplotlib.pyplot as plt
import pickle, json
# from scipy.optimize import fsolve
# from enterprise_extensions import deterministic as det
import scipy.constants as sc
import importlib, inspect
try:
    import healpy as hp
except:
    print('healpy module not found.')

# load spectrum functions from "spectrum.py"
module = importlib.import_module('PTAfast.spectrum')
spec = inspect.getmembers(module, inspect.isfunction)
spec_params = {}
for s_name, s_obj in spec:
    pnames = [*inspect.signature(s_obj).parameters]
    pnames.remove('f')
    spec_params[s_name] = pnames
spec = dict(spec)

class Pulsar:

    def __init__(self, toas, toaerr, theta, phi, pdist=(1., 0.2), freqs=[1400], \
                 custom_noisedict=None, custom_model=None, tm_params=None, backends=['backend'], ephem=None):

        self.nepochs = len(toas)
        self.toas = np.repeat(toas, len(backends))
        self.toaerrs = toaerr * np.ones(len(self.toas))
        self.residuals = np.zeros(len(self.toas))
        self.residuals_gauss = np.zeros(len(self.toas))
        self.Tspan = np.amax(self.toas) - np.amin(self.toas)
        if custom_model is None:
            self.custom_model = {'RN':30, 'DM':100, 'Sv':None}
        else:
            self.custom_model = custom_model
        self.signal_model = {}
        self.flags = {}
        self.flags['pta'] = ['FAKE'] * len(self.toas)
        # self.freqs = np.tile(freqs, self.nepochs)
        # self.backend_flags = np.random.choice(backends, size=len(self.toas), replace=True)
        # self.backend_flags = np.array([bf+'.'+str(int(f)) for bf, f in zip(self.backend_flags, self.freqs)])
        self.freqs, self.backend_flags = self.get_freqs_and_backends(freqs, backends)
        self.backends = np.unique(self.backend_flags)
        self.freqs = abs(self.freqs + np.random.normal(scale=10, size=len(self.freqs)))
        self.theta = theta
        self.phi = phi
        self.pos = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
        if ephem is not None:
            self.ephem = ephem
            self.planetssb = ephem.get_planet_ssb(self.toas)
            self.pos_t = np.tile(self.pos, (len(self.toas), 1))
        else:
            self.planetssb = None
            self.pos_t = None
        self.pdist = pdist
        self.name = self.get_psrname()
        self.init_tm_pars(tm_params)
        self.make_Mmat()
        self.fitpars = [*self.tm_pars]
        self.init_noisedict(custom_noisedict)

    def get_freqs_and_backends(self, freqs, backends):

        b_freqs = []
        backend_flags = np.tile(backends, self.nepochs)
        for i in range(len(backend_flags)):
            try:
                b_freqs.append(float(backend_flags[i].split('.')[-1]))
            except:
                obs_freqs = np.random.choice(freqs)
                backend_flags[i] = backend_flags[i] + '.' + str(int(obs_freqs))
                b_freqs.append(obs_freqs)
        return np.array(b_freqs), backend_flags

    def init_noisedict(self, custom_noisedict=None):

        if custom_noisedict is None:
            custom_noisedict = {}
            noisedict = {}
            for backend in self.backends:
                noisedict[self.name+'_'+backend+'_efac'] = 1.
                noisedict[self.name+'_'+backend+'_log10_tnequad'] = -8.
                noisedict[self.name+'_'+backend+'_log10_t2equad'] = -8.
                noisedict[self.name+'_'+backend+'_log10_ecorr'] = -8.
            self.noisedict = noisedict
        elif np.any([self.name in key for key in [*custom_noisedict]]):
            keys = [*custom_noisedict]
            noisedict = {}
            for key in keys:
                if self.name in key:
                    noisedict[key] = custom_noisedict[key]
            self.noisedict = noisedict
        elif np.all([backend+'_efac' in [*custom_noisedict] for backend in self.backends]):
            noisedict = {}
            for backend in self.backends:
                noisedict[self.name+'_'+backend+'_efac'] = custom_noisedict[backend+'_efac']
                noisedict[self.name+'_'+backend+'_log10_tnequad'] = custom_noisedict[backend+'_log10_tnequad']
                try:
                    noisedict[self.name+'_'+backend+'_log10_t2equad'] = custom_noisedict[backend+'_log10_t2equad']
                except:
                    continue
                try:
                    noisedict[self.name+'_'+backend+'_log10_ecorr'] = custom_noisedict[backend+'_log10_ecorr']
                except:
                    continue
            self.noisedict = noisedict
        else:
            noisedict = {}
            for backend in self.backends:
                noisedict[self.name+'_'+backend+'_efac'] = custom_noisedict['efac']
                noisedict[self.name+'_'+backend+'_log10_tnequad'] = custom_noisedict['log10_tnequad']
                try:
                    noisedict[self.name+'_'+backend+'_log10_t2equad'] = custom_noisedict['log10_t2equad']
                except:
                    continue
                try:
                    noisedict[self.name+'_'+backend+'_log10_ecorr'] = custom_noisedict[backend+'_log10_ecorr']
                except:
                    continue
            self.noisedict = noisedict
        if np.any(['red_noise' in key for key in [*custom_noisedict]]):
            try:
                key_amp = self.name+'_red_noise_log10_A' if self.name+'_red_noise_log10_A' in [*custom_noisedict] else 'red_noise_log10_A'
                key_gam = self.name+'_red_noise_gamma' if self.name+'_red_noise_gamma' in [*custom_noisedict] else 'red_noise_gamma'
                noisedict[self.name+'_red_noise_log10_A'] = custom_noisedict[key_amp]
                noisedict[self.name+'_red_noise_gamma'] = custom_noisedict[key_gam]
            except:
                pass
        if np.any(['dm_gp' in key for key in [*custom_noisedict]]):
            try:
                key_amp = self.name+'_dm_gp_log10_A' if self.name+'_dm_gp_log10_A' in [*custom_noisedict] else 'dm_gp_log10_A'
                key_gam = self.name+'_dm_gp_gamma' if self.name+'_dm_gp_gamma' in [*custom_noisedict] else 'dm_gp_gamma'
                noisedict[self.name+'_dm_gp_log10_A'] = custom_noisedict[key_amp]
                noisedict[self.name+'_dm_gp_gamma'] = custom_noisedict[key_gam]
            except:
                pass
        if np.any(['chrom_gp' in key for key in [*custom_noisedict]]):
            try:
                key_amp = self.name+'_chrom_gp_log10_A' if self.name+'_chrom_gp_log10_A' in [*custom_noisedict] else 'chrom_gp_log10_A'
                key_gam = self.name+'_chrom_gp_gamma' if self.name+'_chrom_gp_gamma' in [*custom_noisedict] else 'chrom_gp_gamma'
                noisedict[self.name+'_chrom_gp_log10_A'] = custom_noisedict[key_amp]
                noisedict[self.name+'_chrom_gp_gamma'] = custom_noisedict[key_gam]
            except:
                pass
        
        self.noisedict = noisedict

    def init_tm_pars(self, timing_model):

        self.tm_pars = {}
        self.tm_pars['F0'] = (200, 1e-13)
        self.tm_pars['F1'] = (0., 1e-20)
        self.tm_pars['DM'] = (0., 5e-4)
        self.tm_pars['DM1'] = (0., 1e-4)
        self.tm_pars['DM2'] = (0., 1e-5)
        self.tm_pars['ELONG'] = (0., 1e-5)
        self.tm_pars['ELAT'] = (0., 1e-5)
        if timing_model is not None:
            self.tm_pars.update(timing_model)

    def make_Mmat(self, t0=0.):

        npar = len([*self.tm_pars]) + 1
        self.Mmat = np.zeros((len(self.toas), npar))
        self.Mmat[:, 0] = np.ones(len(self.toas))
        self.Mmat[:, 1] = -(self.toas - t0) / self.tm_pars['F0'][0]
        self.Mmat[:, 2] = -0.5 * (self.toas - t0)**2 / self.tm_pars['F0'][0]
        self.Mmat[:, 3] = 1 / self.freqs**2
        self.Mmat[:, 4] = (self.toas - t0) / self.freqs**2 / self.tm_pars['F0'][0]
        self.Mmat[:, 5] = 0.5 * (self.toas - t0)**2 / self.freqs**2 / self.tm_pars['F0'][0]
        self.Mmat[:, 6] = np.cos(2*np.pi/sc.Julian_year * (self.toas - t0))
        self.Mmat[:, 7] = np.sin(2*np.pi/sc.Julian_year * (self.toas - t0))

    def update_position(self, theta, phi, update_name=False):
        
        self.theta = theta
        self.phi = phi
        self.pos = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
        if update_name:
            self.name = self.get_psrname()

    def update_noisedict(self, prefix, dict_vals):

        params = {}
        for key in [*dict_vals]:
            params[prefix+'_'+key] = dict_vals[key]
        self.noisedict.update(params)

    def make_ideal(self):

        # set residuals to zero and clean signal model dict

        self.residuals = np.zeros(len(self.toas))
        self.residuals_gauss = np.zeros(len(self.toas))
        for signal in [*self.signal_model]:
            self.signal_model.pop(signal)
            for key in [*self.noisedict]:
                if signal in key:
                    self.noisedict.pop(key)

    def add_white_noise(self, add_ecorr=False, randomize=False, \
                        wn_prior_dict=None):

        if randomize:
            for key in [*self.noisedict]:
                if wn_prior_dict is None: # fpta defaults
                    if 'efac' in key:
                        self.noisedict[key] = np.random.uniform(0.5, 2.5)
                    if 'equad' in key:
                        self.noisedict[key] = np.random.uniform(-8., -5.)
                    if add_ecorr and 'ecorr' in key:
                        self.noisedict[key] = np.random.uniform(-10., -7.)
                else:   
                    if 'efac' in key:
                        wn_prior=wn_prior_dict['efac']
                        # self.noisedict[key] = np.random.uniform(0.8, 1.1)
                        self.noisedict[key] = np.random.uniform(wn_prior['min'], wn_prior['max'])
                    if 'equad' in key:
                        wn_prior=wn_prior_dict['equad']
                        # self.noisedict[key] = np.random.uniform(-8., -6.)
                        self.noisedict[key] = np.random.uniform(wn_prior['min'], wn_prior['max'])
                    if add_ecorr and 'ecorr' in key:
                        wn_prior=wn_prior_dict['ecorr']
                        # self.noisedict[key] = np.random.uniform(-8., -6.)
                        self.noisedict[key] = np.random.uniform(wn_prior['min'], wn_prior['max'])
        if self.backends is None:
            toaerrs2 = self.noisedict[self.name+'_efac']**2 * self.toaerrs**2 + 10**(2*self.noisedict[self.name+'_log10_tnequad'])
        else:
            toaerrs2 = np.zeros(len(self.toaerrs))
            for backend in self.backends:
                mask_backend = self.backend_flags == backend
                toaerrs2[mask_backend] = self.noisedict[self.name+'_'+backend+'_efac']**2 * self.toaerrs[mask_backend]**2 + 10**(2*self.noisedict[self.name+'_'+backend+'_log10_tnequad'])
        
        if add_ecorr:
            for backend in self.backends:
                quant_idx = self.quantise_ecorr(backends=[backend])
                for q_i in quant_idx:
                    if len(q_i) < 2:
                        self.residuals[q_i] += np.random.normal(scale=toaerrs2[q_i]**0.5)
                        self.residuals_gauss[q_i] += np.random.normal(scale=toaerrs2[q_i]**0.5)
                    else:
                        white_block = np.ones((len(q_i), len(q_i))) * 10**self.noisedict[self.name+'_'+backend+'_log10_ecorr']
                        white_block = np.fill_diagonal(white_block, np.diag(white_block) + toaerrs2[q_i])
                        self.residuals[q_i] += np.random.multivariate_normal(mean=np.zeros(len(q_i)), cov=white_block)
                        self.residuals_gauss[q_i] += np.random.multivariate_normal(mean=np.zeros(len(q_i)), cov=white_block)
        else:
            self.residuals += np.random.normal(scale=toaerrs2**0.5)
            self.residuals_gauss += np.random.normal(scale=toaerrs2**0.5)

    def quantise_ecorr(self, dt=1, backends=None):

        if backends is None:
            backends = self.backends

        times = self.toas - self.toas[0]
        quantised_idx = []
        dt *= 24 * 3600
        for backend in backends:
            backend_mask = self.backend_flags == backend
            b_idx = np.arange(len(times))[backend_mask]
            t0 = times[b_idx[0]]
            q_i = [b_idx[0]]
            for n in b_idx[1:]:
                if times[n] - t0 < dt:
                    q_i.append(n)
                else:
                    t0 = times[n]
                    quantised_idx.append(np.array(q_i))
                    q_i = [n]
        
        return quantised_idx

        
        # self.residuals[mask] += 10**(2*self.noisedict[self.name+'_'+backend+'_ecorr']) * np.random.normal()

    def add_red_noise(self, spectrum='powerlaw', f_psd=None, **kwargs):

        rn_components = self.custom_model['RN']
        if rn_components is not None:

            if f_psd is None:
                f_psd = np.arange(1, rn_components+1) / self.Tspan

            if 'red_noise' in self.signal_model:
                self.residuals -= self.reconstruct_signal(['red_noise'])
                self.residuals_gauss -= self.reconstruct_signal(['red_noise'])

            # if spectrum is 'custom':
            if spectrum == 'custom':
                psd = kwargs['custom_psd']
            elif spectrum in [*spec]:
                if len(kwargs) == 0:
                    try:
                        kwargs = {pname : self.noisedict[self.name+'_red_noise_'+pname] for pname in spec_params[spectrum]}
                    except:
                        print('PSD parameters must be in noisedict or parsed as input.')
                        return
                psd = spec[spectrum](f_psd, **kwargs)
                self.update_noisedict(self.name+'_red_noise', kwargs)

                self.add_time_correlated_noise(signal='red_noise', spectrum=spectrum, idx=0., psd=psd, f_psd=f_psd)

    def add_dm_noise(self, spectrum='powerlaw', f_psd=None, **kwargs):

        dm_components = self.custom_model['DM']
        if dm_components is not None:
            
            if f_psd is None:
                f_psd = np.arange(1, dm_components+1) / self.Tspan

            if 'dm_gp' in self.signal_model:
                self.residuals -= self.reconstruct_signal(['dm_gp'])
                self.residuals_gauss -= self.reconstruct_signal(['dm_gp'])

            # if spectrum is 'custom':
            if spectrum == 'custom':
                psd = kwargs['custom_psd']
            elif spectrum in [*spec]:
                if len(kwargs) == 0:
                    try:
                        kwargs = {pname : self.noisedict[self.name+'_dm_gp_'+pname] for pname in spec_params[spectrum]}
                    except:
                        print('PSD parameters must be in noisedict or parsed as input.')
                        return
                psd = spec[spectrum](f_psd, **kwargs)
                self.update_noisedict(self.name+'_dm_gp', kwargs)

            self.add_time_correlated_noise(signal='dm_gp', spectrum=spectrum, idx=2., psd=psd, f_psd=f_psd)

    def add_chromatic_noise(self, spectrum='powerlaw', f_psd=None, **kwargs):

        sv_components = self.custom_model['Sv']
        if sv_components is not None:
            
            if f_psd is None:
                f_psd = np.arange(1, sv_components+1) / self.Tspan

            if 'chrom_gp' in self.signal_model:
                self.residuals -= self.reconstruct_signal(['chrom_gp'])
                self.residuals_gauss -= self.reconstruct_signal(['chrom_gp'])

            # if spectrum is 'custom':
            if spectrum == 'custom':
                psd = kwargs['custom_psd']
            elif spectrum in [*spec]:
                if len(kwargs) == 0:
                    try:
                        kwargs = {pname : self.noisedict[self.name+'_chrom_gp_'+pname] for pname in spec_params[spectrum]}
                    except:
                        print('PSD parameters must be in noisedict or parsed as input.')
                        return
                psd = spec[spectrum](f_psd, **kwargs)
                self.update_noisedict(self.name+'_chrom_gp', kwargs)

            self.add_time_correlated_noise(signal='chrom_gp', spectrum=spectrum, idx=4, psd=psd, f_psd=f_psd)

    def add_system_noise(self, backend=None, components=30, spectrum='powerlaw', f_psd=None, **kwargs):

        assert backend is not None, '"backend" name where system noise is injected must be given'

        if f_psd is None:
            f_psd = np.arange(1, components+1) / self.Tspan

        if 'system_noise_'+str(backend) in self.signal_model:
            self.residuals -= self.reconstruct_signal(['system_noise_'+str(backend)])
            self.residuals_gauss -= self.reconstruct_signal(['system_noise_'+str(backend)])

        # if spectrum is 'custom':
        if spectrum == 'custom':
            psd = kwargs['custom_psd']
        elif spectrum in [*spec]:
            if len(kwargs) == 0:
                try:
                    kwargs = {pname : self.noisedict[self.name+'_system_noise_'+str(backend)+'_'+pname] for pname in spec_params[spectrum]}
                except:
                    print('PSD parameters must be in noisedict or parsed as input.')
                    return
            psd = spec[spectrum](f_psd, kwargs)
            self.update_noisedict(self.name+'_system_noise_'+str(backend), kwargs)

        self.add_time_correlated_noise(signal='system_noise_'+str(backend), idx=0., backend=backend, psd=psd, f_psd=f_psd)

    def add_time_correlated_noise(self, signal='', spectrum='powerlaw', psd=None, f_psd=None, idx=0, freqf=1400, backend=None):

        # generate time correlated noise with given PSD and chromatic index

        if backend is not None:
            signal = backend + '_' + signal
            mask = self.backend_flags == backend
            if not np.any(mask):
                print(backend, 'not found in backend_flags.')
                return
        else:
            mask = np.ones(len(self.toas), dtype='bool')

        df = np.diff(np.append(0., f_psd))
        assert len(psd) == len(f_psd), '"psd" and "f_psd" must be same length. The frequencies "f_psd" correspond to the frequencies where the "psd" is evaluated.'
        psd = np.repeat(psd, 2)

        coeffs = np.random.normal(loc=0., scale=np.sqrt(psd))

        # save noise properties in signal model
        self.signal_model[signal] = {}
        self.signal_model[signal]['spectrum'] = spectrum
        self.signal_model[signal]['f'] = f_psd
        self.signal_model[signal]['psd'] = psd[::2]
        self.signal_model[signal]['fourier'] = np.vstack((coeffs[::2] / df**0.5, coeffs[1::2] / df**0.5))
        self.signal_model[signal]['nbin'] = len(f_psd)
        self.signal_model[signal]['idx'] = idx
        
        for i in range(len(f_psd)):
            self.residuals[mask] += (freqf/self.freqs)**idx * df[i]**0.5 * coeffs[2*i] * np.cos(2*np.pi*f_psd[i]*self.toas[mask])
            self.residuals[mask] += (freqf/self.freqs)**idx * df[i]**0.5 * coeffs[2*i+1] * np.sin(2*np.pi*f_psd[i]*self.toas[mask])

            self.residuals_gauss[mask] += (freqf/self.freqs)**idx * df[i]**0.5 * coeffs[2*i] * np.cos(2*np.pi*f_psd[i]*self.toas[mask])
            self.residuals_gauss[mask] += (freqf/self.freqs)**idx * df[i]**0.5 * coeffs[2*i+1] * np.sin(2*np.pi*f_psd[i]*self.toas[mask])

    def make_time_correlated_noise_cov(self, signal='', freqf=1400):

        # returns covariance matrix of time correlated noise with given PSD and chromatic index

        if 'system_noise' in signal:
            backend = signal.split('system_noise_')[1]
        else:
            backend = None

        if backend is not None:
            signal = backend + '_' + signal
            mask = self.backend_flags == backend
            if not np.any(mask):
                print(backend, 'not found in backend_flags.')
                return
        else:
            mask = np.ones(len(self.toas), dtype='bool')

        # save noise properties in signal model
        f = self.signal_model[signal]['f']
        psd = self.signal_model[signal]['psd']
        components = self.signal_model[signal]['nbin']
        idx = self.signal_model[signal]['idx']

        df = np.diff(np.append(0, f))
        psd = np.repeat(psd * df, 2)
        basis = np.zeros((len(self.toas[mask]), 2*components))
        for i in range(components):
            basis[:, 2*i] = (freqf/self.freqs)**idx * np.cos(2*np.pi*f[i]*self.toas[mask])
            basis[:, 2*i+1] = (freqf/self.freqs)**idx * np.sin(2*np.pi*f[i]*self.toas[mask])
        cov = np.dot(basis, np.dot(np.diag(psd), basis.T))
        return cov
        
    # def add_cgw(self, costheta, phi, cosinc, log10_mc, log10_fgw, log10_h, phase0, psi, psrterm=False):

    #     # add continuous gravitational wave from circular black hole binary

    #     if 'cgw' in self.signal_model:
    #         ncgw = len(self.signal_model['cgw'])
    #     else:
    #         self.signal_model['cgw'] = {}
    #         ncgw = 0
        
    #     self.signal_model['cgw'][str(ncgw)] = {'costheta':costheta, 'phi':phi, 'cosinc':cosinc,
    #                                             'log10_mc':log10_mc, 'log10_fgw':log10_fgw, 'log10_h':log10_h,
    #                                             'phase0':phase0, 'psi':psi, 'psrterm':psrterm}

    #     cgw = det.cw_delay(self.toas, self.pos, self.pdist,
    #                         cos_gwtheta=costheta, gwphi=phi,
    #                         cos_inc=cosinc, log10_mc=log10_mc, 
    #                         log10_fgw=log10_fgw, evolve=True,
    #                         log10_h=log10_h, phase0=phase0, 
    #                         psi=psi, psrTerm=psrterm)
    #     self.residuals += cgw
    #     self.residuals_gauss += cgw

    def add_deterministic(self, waveform, **kwargs):

        fname = waveform.__name__
        if fname in self.signal_model:
            ndet = len(self.signal_model[fname])
        else:
            self.signal_model[fname] = {}
            ndet = 0

        self.signal_model[fname][str(ndet)] = kwargs

        self.residuals += waveform(toas=self.toas, **kwargs)
        self.residuals_gauss += waveform(toas=self.toas, **kwargs)


    def radec_to_thetaphi(ra, dec):

        # RA in format : [H, M]
        # dec in format : [deg, arcmin]

        theta = np.pi/2 -  np.pi/180 * (dec[0] + dec[1]/60)
        phi = 2*np.pi * (ra[0] + ra[1]/60) / 24
        return theta, phi
    
    def thetaphi_to_radec(theta, phi):

        # theta angle
        # phi angle
        DEC = (theta - np.pi/2) * 180 / np.pi
        dec = [int(np.floor(DEC)), int((DEC-np.floor(DEC))*60)]
        RA = phi * 24 / (2*np.pi)
        ra = [int(np.floor(RA)), int((RA-np.floor(RA))*60)]
        return ra, dec

    def get_psrname(self):

        # RA
        h = int(24*self.phi/(2*np.pi))
        m = int((24*self.phi/(2*np.pi) - h) * 60)
        h = '0'+str(h) if len(str(h)) < 2 else str(h)
        m = '0'+str(m) if len(str(m)) < 2 else str(m)
        # DEC
        dec = round(180 * (np.pi/2 - self.theta) / np.pi, 2)
        sign = '+' if dec >= 0 else '-'
        decl, decr = str(abs(dec)).split('.')
        decl = '0'+str(decl) if len(str(decl)) < 2 else str(decl)
        decr = '0'+str(decr) if len(str(decr)) < 2 else str(decr)

        return 'J'+h+m+sign+decl+decr
    
    def make_noise_covariance_matrix(self):

        # make total noise covariance matrix

        if self.backends is None:
            toaerrs = np.sqrt(self.noisedict[self.name+'_efac']**2 * self.toaerrs**2 + 10**(2*self.noisedict[self.name+'_log10_tnequad']))
        else:
            toaerrs = np.zeros(len(self.toas))
            for backend in self.backends:
                mask_backend = self.backend_flags == backend
                toaerrs[mask_backend] = np.sqrt(self.noisedict[self.name+'_'+backend+'_efac']**2 * self.toaerrs[mask_backend]**2 + 10**(2*self.noisedict[self.name+'_'+backend+'_log10_tnequad']))
        white_cov = toaerrs**2

        red_cov = np.zeros((len(self.toas), len(self.toas)))
        if self.custom_model['RN'] is not None:
            red_cov += self.make_time_correlated_noise_cov(signal='red_noise')
        if self.custom_model['DM'] is not None:
            red_cov += self.make_time_correlated_noise_cov(signal='dm_gp')
        if self.custom_model['Sv'] is not None:
            red_cov += self.make_time_correlated_noise_cov(signal='chrom_gp')
        return white_cov, red_cov
    
    def draw_noise_model(self, residuals=None):
        
        white_cov, red_cov = self.make_noise_covariance_matrix()
        cov = np.diag(white_cov) + red_cov
        if residuals is None:
            resids = np.random.multivariate_normal(mean=np.zeros(len(self.toas)), cov=cov)
        else:
            inv_cov = np.linalg.inv(cov)
            resids = np.dot(red_cov.T, np.dot(inv_cov, residuals))
        return resids
    
    def reconstruct_signal(self, signals=None, freqf=1400):

        # reconstruct time domain realisation of injected noises and signals

        if signals is None:
            signals = [*self.signal_model]
        sig = np.zeros(len(self.toas))
        for signal in signals:
            # if signal == 'cgw':
            #     for ncgw in len(self.signal_model['cgw']):
            #         sig += det.cw_delay(self.toas, self.pos, self.pdist,
            #                             **self.signal_model['cgw'][str(ncgw)])
            if (signal in ['red_noise', 'dm_gp', 'chrom_gp']) or ('common' in signal):
                f = self.signal_model[signal]['f']
                idx = self.signal_model[signal]['idx']
                df = np.diff(np.append(0., f))
                c = self.signal_model[signal]['fourier']
                for c_k, f_k, df_k in zip(c.T, f, df):
                    sig += df_k * c_k[0] * (freqf/self.freqs)**idx * np.cos(2*np.pi*f_k * self.toas)
                    sig += df_k * c_k[1] * (freqf/self.freqs)**idx * np.sin(2*np.pi*f_k * self.toas)
            if 'system_noise' in signal:
                backend = signal.split('system_noise_')[1]
                mask = self.backend_flags == backend
                f = self.signal_model[signal]['f']
                df = np.diff(np.append(0., f))
                c = self.signal_model[signal]['fourier']
                for c_k, f_k, df_k in zip(c.T, f, df):
                    sig[mask] += df_k * c_k[0] * np.cos(2*np.pi*f_k * self.toas[mask])
                    sig[mask] += df_k * c_k[1] * np.sin(2*np.pi*f_k * self.toas[mask])
        return sig
    
    def remove_signal(self, signals=None, freqf=1400):

        # remove signal from residuals, signal model and noisedict

        res = self.reconstruct_signal(signals, freqf=freqf)
        self.residuals -= res
        self.residuals_gauss -= res
        for signal in signals:
            self.signal_model.pop(signal)
            for key in [*self.noisedict]:
                if signal in key:
                    self.noisedict.pop(key)


def make_gauss_array(npsrs=25, Tobs=None, ntoas=None, gaps=True, toaerr=None, pdist=None, freqs=[1400], isotropic=False, backends=None, noisedict=None, custom_model=None, ephem=None, xz_pulsars=False):

    # pulsar sky positions
    if xz_pulsars:
        # place all pulsars on the xz-plane: phi = 0 or phi = pi
        costhetas = np.random.uniform(-1., 1., size=npsrs)
        # Alternate phi between 0 and pi for variety, or just use 0
        # phis = np.zeros(npsrs)
        # phis either 0 or pi
        phis = np.pi * np.random.randint(0, 2, size=npsrs)
        # Optionally: phis = np.pi * np.random.randint(0, 2, size=npsrs)
    elif isotropic:
        # Fibonacci sequence on sphere
        i = np.arange(0, npsrs, dtype=float) + 0.5
        golden_ratio = (1 + 5**0.5)/2
        costhetas = 1 - 2*i/npsrs
        phis = np.mod(2 * np.pi * i / golden_ratio, 2*np.pi)
    else:
        costhetas = np.random.uniform(-1., 1., size=npsrs)
        phis = np.random.uniform(0., 2*np.pi, size=npsrs)

    # Observation time for each pulsar
    if Tobs is None:
        Tobs = np.random.uniform(10, 20, size=npsrs)
    elif isinstance(Tobs, float) or isinstance(Tobs, int):
        Tobs = Tobs * np.ones(npsrs)

    # Number of TOAs for each pulsar
    yr = 365.25*24*3600
    if ntoas is None:
        cadence = 7 * 24*3600 # days
        # draw F0 and correct cadence wrt F0
        F0 = np.random.uniform(200, 300, size=npsrs)
        d_cadence = (F0 * cadence - np.floor(F0 * cadence )) / F0
        cadence = cadence - d_cadence
        ntoas = np.int32(Tobs * 365.25 * 24 * 3600 / cadence)
    elif isinstance(ntoas, float) or isinstance(ntoas, int):
        F0 = 200 * np.ones(npsrs)
        ntoas = np.int32(ntoas * np.ones(npsrs))
        cadence = Tobs * yr / (ntoas - 1)

    # Init TOAs from latest observation time
    Tmax = np.amax(Tobs)

    # Make unevenly sampled TOAs if gaps is True
    if gaps:
        gap_odds = [True, True, True, False] # one out of five
        keep = [np.random.choice(gap_odds, size=ntoa) for ntoa in ntoas]
        toas = [(Tmax - Tobs[i])*yr + np.arange(1, ntoas[i]+1)*cadence[i] for i in range(npsrs)]
        toas = [toas[i][keep[i]] for i in range(npsrs)]
    else:
        toas = [(Tmax - Tobs[i])*yr + np.arange(1, ntoas[i]+1)*cadence[i] for i in range(npsrs)]
    if toaerr is None:
        toaerr = np.power(10, np.random.uniform(-7., -5., size=npsrs))
    elif isinstance(toaerr, float):
        toaerr = toaerr * np.ones(npsrs)

    # Init pulsar distances
    if pdist is None:
        dists = np.random.uniform(0.5, 1.5, size=npsrs)
        pdist = [[dist, 0.2*dist] for dist in dists]
    elif isinstance(pdist, float):
        pdist = [[pdist, 0.2*pdist]] * npsrs

    # Init backends
    if backends is None:
        backends = []
        for _ in range(npsrs):
            n_backends = np.random.randint(1, 3)
            backends.append(['backend_'+str(k) for k in range(n_backends)])
    elif isinstance(backends, str):
        backends = [[backends]] * npsrs
    elif isinstance(backends, list):
        if not isinstance(backends[0], list):
            backends = [backends] * npsrs
    
    # Init noise properties


    assert (len(Tobs) == npsrs), '"Tobs" must be same size as "npsrs"'
    assert (len(ntoas) == npsrs), '"ntoas" must be same size as "npsrs"'
    assert (len(toaerr) == npsrs), '"toaerr" must be same size as "npsrs"'
    assert (len(pdist) == npsrs), '"pdist" must be same size as "npsrs"'
    assert (len(backends) == npsrs), '"backends" must be same size as "npsrs"'

    # Create pulsars and add noises
    psrs = []
    for i in range(npsrs):
        if custom_model is None:
            custom_model = None
        psr = Pulsar(toas[i], toaerr[i], np.arccos(costhetas[i]), phis[i], pdist[i], freqs=freqs, backends=backends[i], custom_noisedict=noisedict, custom_model=custom_model, tm_params={'F0':(F0[i], np.random.uniform(1e-13, 1e-12))}, ephem=ephem)
        print('Creating psr', psr.name)
        psr.add_white_noise()
        try:
            psr.add_red_noise(spectrum='powerlaw', log10_A=psr.noisedict[psr.name+'_red_noise_log10_A'], gamma=psr.noisedict[psr.name+'_red_noise_gamma'])
        except:
            psr.add_red_noise(spectrum='powerlaw', log10_A=np.random.uniform(-17., -13), gamma=np.random.uniform(1, 5))
        
        try:
            psr.add_dm_noise(spectrum='powerlaw', log10_A=psr.noisedict[psr.name+'_dm_gp_log10_A'], gamma=psr.noisedict[psr.name+'_dm_gp_gamma'])
        except:
            psr.add_dm_noise(spectrum='powerlaw', log10_A=np.random.uniform(-17., -13), gamma=np.random.uniform(1, 5))
        
        try:
            psr.add_chromatic_noise(spectrum='powerlaw', log10_A=psr.noisedict[psr.name+'_chrom_gp_log10_A'], gamma=psr.noisedict[psr.name+'_chrom_gp_gamma'])
        except:
            psr.add_chromatic_noise(spectrum='powerlaw', log10_A=np.random.uniform(-17., -13), gamma=np.random.uniform(1, 5))
        psrs.append(psr)

    return psrs

# Plot sky positions of pulsars
def plot_pta(psrs, plot_name=False, alpha=0.7, color='r'):

    ax = plt.axes(projection='mollweide')
    ax.grid(True, **{'alpha':0.25})
    plt.xticks(np.pi - np.linspace(0., 2*np.pi, 5), ['0h', '6h', '12h', '18h', '24h'], fontsize=14)
    plt.yticks(fontsize=14)
    for psr in psrs:
        s = 50 * (10**(-6) / np.mean(psr.toaerrs))
        plt.scatter(np.pi - np.array(psr.phi), np.pi/2 - np.array(psr.theta), \
                    marker=(5, 1), s=s, alpha=alpha, color=color)
        if plot_name:
            plt.annotate(psr.name, (np.pi - psr.phi + 0.05, np.pi/2 - psr.theta - 0.1), color='k', fontsize=10)
    plt.show()

# Copy existing array
def copy_array(psrs, custom_noisedict, custom_models=None):

    if custom_models is None:
        custom_models = {}
        for psr in psrs:
            custom_models[psr.name] = None

    fake_psrs = []
    for psr in psrs:
        fake_psr = Pulsar(psr.toas, 10**(-6), psr.theta, phi=psr.phi, pdist=1., backends=np.unique(psr.backend_flags), custom_model=custom_models[psr.name])
        fake_psr.name = psr.name
        fake_psr.toas = psr.toas
        fake_psr.toaerrs = psr.toaerrs
        fake_psr.residuals = psr.residuals
        fake_psr.Mmat = psr.Mmat
        fake_psr.fitpars = psr.fitpars
        fake_psr.pdist = psr.pdist
        fake_psr.backend_flags = psr.backend_flags
        fake_psr.backends = np.unique(psr.backend_flags)
        fake_psr.freqs = psr.freqs
        fake_psr.planetssb = psr.planetssb
        fake_psr.pos_t = psr.pos_t
        fake_psr.init_noisedict(custom_noisedict)
        # OR set fake_psr.noisedict to be custom noisedict
        fake_psrs.append(fake_psr)
    return fake_psrs