class Dowell(PEModel):
    """ Dowell's bulk photoemission model """
    def __init__(self, surface, material, laser, field):
        """ Initialize the Dowell configuration.

        Keyword arguments:
        surface -- the surface of the photocathode.
        material -- the material of the cathode.
        laser -- the incident laser.
        field -- the electric field on the cathode surface.
        """
        super().__init__(surface, material, laser, field)
        # calculate Dowell specified parameters
        self.specs = {}
        self.cal_specs()
        # single point standard statistic
        self.spss = {}
        self.cal_spss()
            
    def cal_specs(self):
        """ Calculate Dowell specified parameters.
        
        The Dowell specified parameter list:
        * lambda_e: electron mean free path in the cathode, [nm]
        * lambda_p: photon mean free path in the cathode, [nm]
        * pm: minimum momentum, [keV/c]
        * pM: maximum momentum, [keV/c]
        * p_max: maximum emitted momentum, [keV/c]
        * E_max: maximum emitted energy, [eV]
        """
        self.specs['lambda_e'] = self.lambda_e()
        self.specs['lambda_p'] = self.lambda_p()
        Ee = const.m_e*const.c**2/(const.e*1e6)
        self.specs['pm'] = np.sqrt(2*Ee*(self.paras['Ef']+self.paras['phi_eff']))
        self.specs['pM'] = np.sqrt(2*Ee*(self.paras['Ef']+self.paras['Ep']))
        self.specs['p_max'] = np.sqrt(self.specs['pM']**2-self.specs['pm']**2)
        self.specs['E_max'] = self.paras['Ep']-self.paras['phi_eff']
    
    def cal_spss(self):
        """ Calculate single point standard statistic parameters.
        
        The single point standard statistic parameter list:
        * px: mean of px, [keV/c]
        * px^2: mean of px^2, [(keV/c)^2]
        * py: mean of py, [keV/c]
        * py^2: mean of py^2, [(keV/c)^2]
        * pz: mean of pz, [keV/c]
        * pz^2: mean of pz^2, [(keV/c)^2]
        * QE: quantum efficiency
        * emittance: sigma_x*sigma_px, [µm.keV/c]
        """
        self.spss['px'] = 0
        self.spss['px^2'] = (self.specs['pM']**2-self.specs['pm']**2)/6.0
        self.spss['py'] = 0
        self.spss['py^2'] = (self.specs['pM']**2-self.specs['pm']**2)/6.0
        x = np.sqrt(self.specs['pM']/self.specs['pm']-1)
        self.spss['pz'] = self.specs['pm']/x**4*((1.0/3*x**5+2.0/3*x**3+x)*np.sqrt(x**2+2)-(x**2+1)\
                                                 *np.arccosh(x**2+1))
        # s['pz'] = 8*np.sqrt(2)/15.0*np.sqrt((self.specs['pM']-self.specs['pm'])*self.specs['pm'])
        self.spss['pz^2'] = (self.specs['pM']-self.specs['pm'])*(self.specs['pM']+3*self.specs['pm'])/6.0 
        self.spss['QE'] = (1-self.reflectivity(0))/(1+self.specs['lambda_p']/self.specs['lambda_e'])/2\
                          *(1+self.paras['Ef']/self.paras['Ep'])*(1-self.specs['pm']/self.specs['pM'])**2
        self.spss['emittance'] = self.laser.sigma_r/np.sqrt(2)*np.sqrt(self.spss['px^2'])
            
    def refresh(self):
        """ Re-calculate all the parameters.
        """
        super().refresh()
        self.cal_specs()
        self.cal_spss()

    def lambda_e(self):
        """ Calculate the mean free path of an electron travel in a bulk of metal.
        
        The unit of mean free path is nm."""
        # unit: nm
        lambda_m = 2.2 # unit: nm
        Em = 8.6 # unit: eV
        return 2*lambda_m*Em**(1.5)/(self.paras['Ep']*np.sqrt(self.paras['phi_eff']))\
               /(1+np.sqrt(self.paras['phi_eff']/self.paras['Ep']))
        
    def lambda_p(self):
        """ Calculate the mean free path of a photon in a bulk of metal.
        
        The unit of mean free path is nm."""
        # unit: nm
        return self.paras['lambda_l']/(4*np.pi*self.paras['refractivity'].imag)
    
    def pdf_spher(self, r, theta, phi):
        """ Dowell's single point momentum pdf in spherical coordinates.

        Keyword arguments:
        r -- the radial distance.
        theta -- the polar angle.
        phi -- the azimuth angle.

        Returns:
        intensity of that point.
        """
        return 1/(np.pi*(self.specs['pM']-self.specs['pm'])**2)*r**3*np.cos(theta)*np.sin(theta)\
               /(np.sqrt(r**2+self.specs['pm']**2)*np.sqrt((r*np.cos(theta))**2+self.specs['pm']**2))
    
    def momentum_samples(self, N=10000):
        """ Get N single point momentum samples that obey the Dowell's momentum pdf.

        Keyword arguments:
        N -- number of samples to be generated. [10000]

        Returns:
        samples -- 3*N array, (px, py, pz).
        """
        samples = []        
        M = 1/np.pi*(1+self.specs['pm']/self.specs['pM'])/self.specs['p_max']
        trans = lambda r, theta, phi: (r*np.sin(theta)*np.cos(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(theta))
        
        for i in range(N):
            while True:
                a = np.random.rand(4)
                a[0] *= self.specs['p_max']
                a[1] *= np.pi/2
                a[2] *= 2*np.pi
                a[3] *= M
                if a[3] <= self.pdf_spher(a[0], a[1], a[2]):
                    samples.append(trans(a[0], a[1], a[2]))
                    break
        return np.array(samples).transpose()
    
    def gen_init_bunch(self, N=10000, roughness=True):
        """ Generate the initial emitted electron bunch.
        
        Note that everytime the Dowell setup changes, the initial bunch distribution will also change!
        Dowell setup includes:
        * laser incident position
        * laser incident angle
        * applied electric field strength
        When using dowell.setup to update the settings above, all the paras will automatically update,
        but if you would like to change laser itself, or surface, or material, etc that is not included in 
        the Dowell setup, you have to call dowell.refresh yourself.

        Keyword arguments:
        N -- number of samples to be generated. [10000]
        roughness -- take account of the roughness or not, choose between 0 and 1.

        Returns:
        samples -- (x0, y0, z0, px0, py0, pz0), each column is a Nl array, Nl <= N.
        """
        x0, y0 = self.laser_samples(N)
        z0 = self.surface.surf((x0, y0))
        Nl = x0.shape[0]
        _px0, _py0, _pz0 = self.momentum_samples(Nl)
        
        if roughness:
            Rx, Ry = self.surface.partial()
            rx = -Rx((x0, y0))*1e-3
            ry = -Ry((x0, y0))*1e-3
            c1, c2, c3 = rx/np.sqrt(rx**2+ry**2+1), ry/np.sqrt(rx**2+ry**2+1), 1.0/np.sqrt(rx**2+ry**2+1)
            a1, a2, a3 = -c2/np.sqrt(c1**2+c2**2), c1/np.sqrt(c1**2+c2**2), np.zeros(c1.shape)
            t = np.sqrt(c1**2+c2**2+((c1**2+c2**2)/c3)**2)
            b1, b2, b3 = c1/t, c2/t, -(c1**2+c2**2)/(c3*t)

            px0 = _px0*a1+_py0*b1+_pz0*c1
            py0 = _px0*a2+_py0*b2+_pz0*c2
            pz0 = _px0*a3+_py0*b3+_pz0*c3

            return x0, y0, z0, px0, py0, pz0
        else:
            return x0, y0, z0, _px0, _py0, _pz0
        
    def slope_effect(self, samples):
        """ Add slope effect to the given bunch samples.

        Keyword arguments:
        samples -- initial bunch samples.

        Returns:
        samples with slope effect.
        """
        x0, y0, z0, _px0, _py0, _pz0 = samples
        
        Rx, Ry = self.surface.partial()
        rx = -Rx((x0, y0))*1e-3
        ry = -Ry((x0, y0))*1e-3
        c1, c2, c3 = rx/np.sqrt(rx**2+ry**2+1), ry/np.sqrt(rx**2+ry**2+1), 1.0/np.sqrt(rx**2+ry**2+1)
        a1, a2, a3 = -c2/np.sqrt(c1**2+c2**2), c1/np.sqrt(c1**2+c2**2), np.zeros(c1.shape)
        t = np.sqrt(c1**2+c2**2+((c1**2+c2**2)/c3)**2)
        b1, b2, b3 = c1/t, c2/t, -(c1**2+c2**2)/(c3*t)

        px0 = _px0*a1+_py0*b1+_pz0*c1
        py0 = _px0*a2+_py0*b2+_pz0*c2
        pz0 = _px0*a3+_py0*b3+_pz0*c3
        
        return x0, y0, z0, px0, py0, pz0
    
    def field_effect(self, samples):
        """ Add field effect to the given bunch samples.

        Keyword arguments:
        samples -- initial bunch samples.

        Returns:
        samples with field effect, note that only transverse components are valid.
        """
        x0, y0, z0, px0, py0, pz0 = samples
        
        E0 = self.paras['E0'] # MV/m
        p0 = const.m_e*const.c**2/const.e*1e-3 # keV/c

        IFx, IFy = self.field.int_field()
        Rx, Ry = self.surface.partial()
        r_x = Rx((x0, y0))*1e-3 # change nm/um to 1
        r_x = Ry((x0, y0))*1e-3
        Ix = np.sqrt(np.pi/2*E0*p0*1e-3)*1e-3*IFx((x0, y0)) # keV/c
        Iy = np.sqrt(np.pi/2*E0*p0*1e-3)*1e-3*IFy((x0, y0))
        
        px = px0-Ix
        py = py0-Iy
        
        return x0, y0, z0, px, py, pz0
    
    def stat_emittance(self, samples, direction='x'):
        """ Stat the emittance of the given samples.

        Keyword arguments:
        samples -- bunch samples.
        direction -- 'x' or 'y', specify emittance of which direction to stat.

        Returns:
        emittance.
        """
        x0, y0, z0, px0, py0, pz0 = samples
        
        if direction == 'x':
            return np.sqrt(np.linalg.det(np.cov(x0, px0, bias=1)))
        elif direction == 'y':
            return np.sqrt(np.linalg.det(np.cov(y0, py0, bias=1)))
        else:
            print('Sorry, emittance for direction {} is not supported!'.format(direction))
            return None
    
    def reflectivity(self, theta=0, polarization='p'):
        """ Calculate the reflectivity of the cathode to incident laser.
        
        Keyword arguments:
        theta -- the incident angle, unit: rad; [0]
        polarization -- the polarization of the laser, 
                        could be 's' or 'p' or a float between 0 and 1 
                        (the percentage of the intensity of p-polarized light).
        """
        return self.material.reflectivity(self.paras['lambda_l'], theta, polarization)
    
    def weight(self, theta=0, polarization='p'):
        """ Calculate the normalized QE.
        
        Keyword arguments:
        theta -- the incident angle, unit: rad; [0]
        polarization -- the polarization of the laser, 
                        could be 's' or 'p' or a float between 0 and 1 
                        (the percentage of the intensity of p-polarized light).
        """
        nfactor = (1-self.reflectivity(0))/(self.specs['lambda_e']+self.specs['lambda_p'])
        return (1-self.reflectivity(theta, polarization))*np.cos(theta)/(self.specs['lambda_e']+\
                                                                         self.specs['lambda_p']*\
                                                                         np.cos(theta))/nfactor
    
    def cal_emittance(self, method='theory', **kwargs):
        """ Calculate the bunch emittance.

        Keyword arguments:
        method -- which kind of emittance to be calculated, choose between 'theory', 'simulation' and '2d'.
            * theory: 3D theoretical emittance
            * simulation: emittance derived from the simulation result
            * 2d: 2D sinusoidal theoretical emittance
        N -- number of samples used in 3D and 2D theoretical emittance calculation.
        **kwargs -- 
            * theory: optional: N [10000] or samples, direction ['x'], kind ['field']
                kind -- 'slope', 'field' or 'init', specify which formula to use.
                if the key is N, then generate the bunch samples
                elif the key is samples, use the given **LASER** samples to calculate emittance
                note that the samples should be samples without roughness
            * simulate: optional: slice [-1], direction ['x']
                calculate the emittance at given step number
                if both paras are not given, calculate the emittance for the last step
            * 2d: a [nm] and p [µm] of the 2D sinusoidal surface, optional: crossterm
                a: amplitude of the surface
                p: period of the surface
                crossterm: if consider the crossterm in the 2d formula

        Returns:
        emittance
        """
        if method == 'theory':
            # definitions
            E0 = self.paras['E0'] # MV/m
            p0 = const.m_e*const.c**2/const.e*1e-3 # keV/c
            try:
                direction = kwargs['direction']
            except KeyError:
                direction = 'x'
            try:
                kind = kwargs['kind']
            except KeyError:
                kind = 'field'
            if 'N' in kwargs.keys():
                N = kwargs['N']
                samples = self.laser_samples(N)
            elif 'samples' in kwargs.keys():
                samples = kwargs['samples']
            else:
                N = 10000
                samples = self.laser_samples(N)
            
            x0, y0 = samples # laser samples! NOT electron samples!
            
            if kind == 'slope':
                if direction == 'x':
                    R = self.surface.partial()[0]
                elif direction == 'y':
                    R = self.surface.partial()[1]
                else:
                    print("'direction' should be 'x' or 'y'!")
                    return
                r = R((x0, y0))*1e-3 # change nm/um to 1
                I = 0
            elif kind == 'field':
                if direction == 'x':
                    R = self.surface.partial()[0]
                    IF = self.field.int_field()[0]
                elif direction == 'y':
                    R = self.surface.partial()[1]
                    IF = self.field.int_field()[1]
                else:
                    print("'direction' should be 'x' or 'y'!")
                    return
                r = R((x0, y0))*1e-3 # change nm/um to 1
                I = np.sqrt(np.pi/2*E0*p0*1e-3)*1e-3*IF((x0, y0)) # keV/c
            elif kind == 'init':
                r = 0
                I = 0
            else:
                print("'kind' should be 'slope', 'field' or 'init'!")
                return
            
            sigma_x = self.laser.sigma_r/np.sqrt(2)
            sigma_p = np.sqrt(self.spss['px^2']+np.mean(r**2)*(self.spss['pz^2']-self.spss['px^2'])+\
                              2*self.spss['pz']*np.mean(r*I)+np.mean(I**2))
            emittance = sigma_x*sigma_p
            
            return emittance
        
        elif method == 'simulation':
            try:
                num = kwargs['slice']
            except KeyError:
                num = -1
            try:
                direction = kwargs['direction']
            except KeyError:
                direction = 'x'
            try:
                y_result = self.y_result
                t_output = self.t_output
                samples = np.split(y_result[num], 6)
                emittance = self.stat_emittance(samples, direction)
                return emittance
            except:
                print("No simulation data!")
                return
            
        elif method == '2d':
            try:
                a = kwargs['a']*1e-9 # m
                lamb = kwargs['p']*1e-6 # m
            except:
                print("Please specify 'a' and/or 'p'!")
                return
            try:
                crossterm = kwargs['crossterm']
            except KeyError:
                crossterm = 1
            
            E0 = self.paras['E0'] # MV/m
            p0 = const.m_e*const.c**2/const.e*1e-3 # keV/c
            
            k = 2*np.pi/lamb # m^-1
            xi = a*k
            
            pC = np.sqrt(p0*np.pi*E0*1e3/(2*k))
            if crossterm:
                eta = np.sqrt(1+xi**2/2*((self.spss['pz^2']+pC**2+2*pC*self.spss['pz'])/self.spss['px^2']-1))
            else:
                eta = np.sqrt(1+xi**2/2*((self.spss['pz^2']+pC**2)/self.spss['px^2']-1))
            emittance = eta*self.spss['emittance'] # µm.keV/c
            
            return emittance
        
        else:
            print("Sorry, emittance type {} is not supported!".format(method))
            
            return None
