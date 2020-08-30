class Material(object):
    """ Cathode material """
    def __init__(self, name, mode='linear', path='', extra=1):
        """ Initialize the material.
        
        Input the refractive index data (and extra work function and Fermi energy data), generate a material.

        Keyword arguments:
        name -- name of the material. Note that the name should also be the name of .ri file (and .json file).
        mode -- interpolation method for the refractivity index data.
        path -- path that contains the .ri file (and .json file).
        extra -- if read the extra infomation of the material.
            if extra is set to 0, then won't read the .json file,
            which contains the work function and Fermi energy of the material.
        """
        object.__init__(self)
        self.name = name
        # read the refractive index, shape: 3*N array
        try:
            ior_filename = os.path.join(path, name + '.ri')
            ri = np.loadtxt(ior_filename).transpose() # shape: 3*N
            ri[0] *= 1e3 # change wavelength unit to nm
            self._mode = mode
            self.ri_data = ri
            self.ri_interp = (interp1d(ri[0], ri[1], mode), interp1d(ri[0], ri[2], mode))
        except IOError:
            print("Sorry, error when reading .ri file for material {}!".format(name))
            self.ior = None
        if extra:
            try:
                paras_filename = os.path.join(path, name + '.json')
                f = open(paras_filename)
                self.paras = json.load(f)
                f.close()
            except IOError:
                print("Sorry, error when reading .json file for material {}!".format(name))
                self.paras = None
                
    @property
    def mode(self):
        """ Interpolation method for the refractivity index data.
        """
        return self._mode

    @mode.setter
    def mode(self, value):
        self._set_interpolation_mode(value)
            
    def _set_interpolation_mode(self, mode):
        """ Set (change) the interpolation method.
        
        Also change the self.ri_interp, of course.

        Keyword arguments:
        mode -- the name of the new interpolation method.
        """
        if mode in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']:
            self._mode = mode
            ri = self.ri_data
            self.ri_interp = (interp1d(ri[0], ri[1], mode), interp1d(ri[0], ri[2]), mode)
        else:
            print("Sorry, '{0}' mode is not supported!".format(mode))
            
    def refractivity(self, lamb):
        """ Get the refractive index for an individual wavelength.
        
        Keyword arguments:
        lambda -- the incident light wavelength, unit: nm.
        
        Returns:
        refractive index, format: a+bj.
        """
        return self.ri_interp[0](lamb)+self.ri_interp[1](lamb)*1j
    
    def permittivity(self, lamb):
        """ Get the permittivity for an individual wavelength.
        
        Keyword arguments:
        lambda -- the incident light wavelength, unit: nm.
        
        Returns:
        permittivity, format: a+bj.
        """
        return self.refractivity(lamb)**2
    
    def reflectivity(self, lamb, theta, polarization='p'):
        """ Get the reflectivity for an individual laser.

        Keyword arguments:
        lambda -- the incident light wavelength, unit: nm;
        theta -- the incident angle, unit: rad;
        polarization -- the polarization of the laser, 
                        could be 's' or 'p' or a float between 0 and 1 
                        (the percentage of the intensity of p-polarized light).
        
        Returns:
        reflectivity
        """
        ep = self.permittivity(lamb)
        if polarization == 'p':
            frac = ep*np.cos(theta)/np.sqrt(ep-np.sin(theta)**2+0j)
        elif polarization == 's':
            frac = np.cos(theta)/np.sqrt(ep-np.sin(theta)**2+0j)
        else:
            eta = polarization
            frac_p = ep*np.cos(theta)/np.sqrt(ep-np.sin(theta)**2+0j)
            rp = np.abs((1-frac_p)/(1+frac_p))**2
            frac_s = np.cos(theta)/np.sqrt(ep-np.sin(theta)**2+0j)
            rs = np.abs((1-frac_s)/(1+frac_s))**2
            return eta*rp+(1-eta)*rs
        return np.abs((1-frac)/(1+frac))**2
    
    def plot_data(self, datatype='refractivity', save=0):
        """ Draw the refractivity or permittivity of the material vs. wavelength.
        
        Keyword arguments:
        datatype -- choose between 'refractivity' and 'permittivity'.
        save -- save the plot or not.
        """
        fig = plt.figure()#figsize=(8, 6))
        ax = fig.add_subplot(111)        
#         ax.grid(True)
        if datatype == 'refractivity':
            ax.set(xlabel='wavelength (nm)', ylabel='refractive index (1)')#, 
#                    title='Refractive Index of '+str(self.name))
            x = self.ri_data[0]
            y = self.refractivity(x)
            ax.loglog(x, y.real, label='Re, ' + str(self.name))
            ax.loglog(x, y.imag, label='Im, ' + str(self.name))
        elif datatype == 'permittivity':            
            ax.set(xlabel='wavelength (nm)', ylabel='relative permittivity (1)')
            #, title='Permittivity of '+str(self.name))
            x = self.ri_data[0]
            y = self.permittivity(x)
            ax.semilogx(x, y.real, label='Re, ' + str(self.name))
            ax.semilogx(x, y.imag, label='Im, ' + str(self.name))
        ax.legend(loc=0)
        if save:
            fig.savefig('ri_data.pdf', bbox_inches='tight')
        plt.show()
        
    def plot_reflectivity(self, lamb, theta=0, polarization='p', save=0):
        """ Draw the reflectivity of the material vs. wavelength / incident angle / polarization ratio.
        
        Keyword arguments:
        lambda -- the incident light wavelength, unit: nm;
        theta -- the incident angle, unit: rad;
        polarization -- the polarization of the laser, 
                        could be 's' or 'p' or a float between 0 and 1 
                        (the percentage of the intensity of p-polarized light).
        save -- save the plot or not.
        """
        fig = plt.figure()#figsize=(8, 6))
        ax = fig.add_subplot(111)        
#         ax.grid(True)
        if isinstance(lamb, np.ndarray):
            ax.set(xlabel='wavelength (nm)', ylabel='reflectivity (1)')#, 
#                    title='Reflectivity of '+str(self.name)+' vs. Wavelength')
            x = lamb
            y = self.reflectivity(lamb, theta, polarization)
            ax.plot(x, y, label='incident angle = {0:.2f} rad\npolarization = {1}'.format(theta, polarization))
        elif isinstance(theta, np.ndarray):
            ax.set(xlabel='incident angle (rad)', ylabel='reflectivity (1)')#, 
#                    title='Reflectivity of '+str(self.name)+' vs. Incident Angle')
            x = theta
            y = self.reflectivity(lamb, theta, polarization)
            ax.plot(x, y, label='wavelength = {0:.2f} nm\npolarization = {1}'.format(lamb, polarization))
        elif isinstance(polarization, np.ndarray):
            ax.set(xlabel='polarization ratio (1)', ylabel='reflectivity (1)')#, 
#                    title='Reflectivity of '+str(self.name)+\
#                    ' vs. Polarization')
            x = polarization
            y = self.reflectivity(lamb, theta, polarization)
            ax.plot(x, y, label='wavelength = {0:.2f} nm\nincident angle = {1:.2f} rad'.format(lamb, theta))
        else:
            print("Sorry, plot type isn't supported!")
        ax.legend(loc=0)
        if save:
            fig.savefig('reflectivity.pdf', bbox_inches='tight')
        plt.show()
