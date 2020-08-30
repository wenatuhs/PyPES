class PEModel(object):
    """ Photoemission models """
    def __init__(self, surface, material, laser, field):
        """ Initialize the photoemission model.

        Keyword arguments:
        surface -- the surface of the photocathode.
        material -- the material of the cathode.
        laser -- the incident laser.
        field -- the electric field on the cathode surface.
        """
        object.__init__(self)
        self.surface = surface
        self.material = material
        self.laser = laser
        self.field = field
        # calculate basic parameters
        self.cal_paras()
        self.paras['pos_in'] = None
        self.paras['theta_in'] = None
        
    def cal_paras(self):
        """ Calculate the basic parameters.
        
        The physical parameter list:
        * phi_w: work function, [eV]
        * Ef: Fermi energy, [eV]
        * lambda_l: laser wavelength, [nm]
        * Ep: photon energy, [eV]
        * E0: applied electric field stength, [MV/m]
        * phi_eff: effective work function, [eV]
        * refractivity: refractivity for the laser wavelength
        * permittivity: permittivity for the laser wavelength
        """
        self.paras = dict(list(self.material.paras.items())+list(self.laser.paras.items())\
                          +list(self.field.paras.items()))
        self.paras['phi_eff'] = self.paras['phi_w']-self.schottky()
        self.paras['refractivity'] = self.material.refractivity(self.paras['lambda_l'])
        self.paras['permittivity'] = self.material.permittivity(self.paras['lambda_l'])
        
    def refresh(self):
        """ Re-calculate the basic parameters.
        """
        self.cal_paras()
                
    def schottky(self):
        """ Calculate the Schottky potential (energy).
        """
        # unit: eV
        return np.sqrt(const.e*self.paras['E0']/(4*const.pi*const.epsilon_0))*1e3
    
    def reflectivity(self, *args, **kwargs):
        """ Calculate the reflectivity of the cathode to incident laser.
        
        Need to be implemented.
        """
        return None
    
    def weight(self, *args, **kwargs):
        """ Calculate the normalized QE.
        
        Need to be implemented.
        """
        return None
    
    def cal_emittance(self, *args, **kwargs):
        """ Calculate the bunch emittance.

        Need to be implemented.

        Returns:
        emittance
        """
        return None
    
    def setup(self, pos_in, theta_in=0, E0=None):
        """ Setup the laser incident position and angle, also the electric field strength.
        
        Keyword arguments:
        pos_in -- laser incident position, unit: um. A 2-element list or a 2-element ndarray.
        theta_in -- laser incident angle, unit: radian.
        E0 -- electric field strength, unit: MV/m.
        
        Automatically refresh after setup. """
        if E0 != None:
            self.field.paras['E0'] = E0
            self.refresh()
        self.paras['pos_in'] = pos_in
        self.paras['theta_in'] = theta_in
        
    def plot_configuration(self, N=10000, area=None, bg=1, save=0, orientation='+'):
        """ Plot the configuration.
    
        A configuration is a combination of surface morphology and the laser distribution.

        Keyword arguments:
        N -- number of laser samples. [10000]
        area -- the plotting area, format is [x0, x1, y0, y1], unit is µm.
            Default is None, which means the whole surface area.
        bg -- if draw the surface morphology, choose between 0 and 1.
        save -- save the plot or not.
        orientation -- choose between '+' and '-'.
            * '+': the surface colorbar is on the right, the weight colorbar is on the top
            * '-': the surface colorbar is on the top, the weight colorbar is on the right
        """
        fig = plt.figure()#figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        if not area:
            area = (0, self.surface.xm, 0, self.surface.ym)
        ax.set(xlabel="x (µm)", ylabel="y (µm)", aspect='equal', 
               xlim=(area[0], area[1]), ylim=(area[2], area[3]))#, title='Experimental Configuration')
        
        if bg:
            if orientation == '+':
                divider = self.surface.plot_surface(area, projection=1, aspect='equal', orientation='v', ax=ax)
            elif orientation == '-':
                divider = self.surface.plot_surface(area, projection=1, aspect='equal', orientation='h', ax=ax)
            else:
                divider = self.surface.plot_surface(area, projection=1, aspect='equal', orientation=None, ax=ax)
        else:
            divider = make_axes_locatable(ax)
        
        #TODO: consider the polarization
        samples = self.laser_samples(N)
        Rx, Ry = self.surface.partial()
        rx = -Rx((samples[0], samples[1]))*1e-3 # change nm/µm to 1
        ry = -Ry((samples[0], samples[1]))*1e-3
        theta = np.arccos(1.0/np.sqrt(rx**2+ry**2+1))
        colors = self.weight(theta)
        if colors is None:
            if orientation == '+':
                self.laser.plot_samples(samples, orientation='h', ax=ax, divider=divider)
            elif orientation == '-':
                self.laser.plot_samples(samples, orientation='v', ax=ax, divider=divider)
            else:
                self.laser.plot_samples(samples, orientation=None, ax=ax, divider=divider)
        else:
            weig = ax.scatter(samples[0], samples[1], c=colors, s=5, marker='.', alpha=0.5, 
                              cmap=cm.coolwarm, edgecolors='none')
            if orientation == '+':
                cax = divider.append_axes("top", size="5%", pad=0.1)
                cbar = fig.colorbar(weig, cax=cax, orientation='horizontal')
                cbar.ax.xaxis.tick_top()
                cbar.ax.set_xlabel('weight (arb. units.)')
                cbar.ax.xaxis.set_label_position('top')
                cbar.set_alpha(1)
                cbar.draw_all()
            elif orientation == '-':
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = fig.colorbar(weig, cax=cax)
                cbar.ax.set_ylabel('weight (arb. units.)')
                cbar.set_alpha(1)
                cbar.draw_all()
            else:
                pass
        
        if save:
            fig.savefig('configuration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def laser_samples(self, N=10000):
        """ Get N laser incident point samples.
        
        Keyword arguments:
        N -- number of samples to be generated. [10000]
        
        Returns:
        samples -- 2*Nl array. Nl <= N. Nl is decided by the laser area and surface area. 
        """
        samples = self.laser.gen_samples(N) + np.array(self.paras['pos_in']).reshape(2, 1)
        f = ((samples[0] > 0) & (samples[1] > 0) & (samples[0] < self.surface.xm) & (samples[1] < self.surface.ym))\
            * np.array([True, True]).reshape(2, -1)
        return samples[f].reshape(2, -1)
        
    def gen_init_bunch(self, N, *args, **kwargs):
        """ Generate the initial emitted electron bunch.
        
        Need to be implemented.

        Keyword arguments:
        N -- number of samples to be generated.

        Returns:
        samples -- (x0, y0, z0, px0, py0, pz0), each column is a Nl array, Nl <= N.
        """
        return None
        
    def simulate(self, init, zi, zf, dz, verbosity=True):
        """ Perform the simulation of the particle dynamics.
        
        Keyword arguments:
        init -- initial electron bunch samples.
        zi -- initial z position. (where the simulation starts) [nm]
        zf -- final z position. (where the simulation ends) [nm]
        dz -- z step size, [nm]
        verbosity -- see the progressbar or not.
        """
        # define constants
        E0 = self.paras['E0'] # MV/m
        p0 = const.m_e*const.c**2/const.e*1e-3 # keV/c
        
        # define motion function
        def surf_dynamics(t, y_vec):
            # t -- z, unit: nm 
            x, y, z, px, py, pz = np.split(y_vec, 6)
            return np.concatenate((px/pz*1e-3, py/pz*1e-3, np.ones(x.shape[0]), \
                                   -p0*1e-6*E0/pz*self.field.Ex((x, y, z)), \
                                   -p0*1e-6*E0/pz*self.field.Ey((x, y, z)), \
                                   -p0*1e-6*E0/pz*self.field.Ez((x, y, z))))
        
        # simulation settings
        t0 = zi # nm
        t_final =  zf # nm
        dt = dz # nm
        total = np.ceil((zf-zi)/float(dz)) # total number of steps
#         gap = int(np.ceil(total/100.0)) # output every gap number of steps
        
        pb = JavaProgressBar(total)
        pb.animate(0, 'initializing...')
        
        # initial condition
        y_vec0 = np.concatenate(init)
        
        # start simulation
        y_result = []
        t_output = []

        backend = "dopri5"

        solver = ode(surf_dynamics)
        solver.set_integrator(backend)
        solver.set_initial_value(y_vec0, t0)

        y_result.append(y_vec0)
        t_output.append(t0)
        
        pb.animate(0, 'simulation starting...')
        if verbosity:
            count = 0
            while solver.successful() and solver.t < t_final:
                solver.integrate(solver.t+dt)
                y_result.append(solver.y)
                t_output.append(solver.t)
                pb.animate(count+1, 'z = {:.2f} nm...'.format(solver.t))
#                 if count % gap == 0:
#                     print("z position: " + str(solver.t) + " nm")
                count += 1
        else:
            while solver.successful() and solver.t < t_final:
                solver.integrate(solver.t+dt)
                y_result.append(solver.y)
                t_output.append(solver.t)
                
        self.y_result = np.array(y_result)
        self.t_output = np.array(t_output)
        if verbosity:
            pb.animate(total, 'done.', 1)
        
    def gen_animation(self, filename='animation'):
        """ Generate the animation of the beam dynamics.

        Keyword arguments:
        filename -- the filename of the animation.
        """
        flag = 0
        
        try:
            y_result = self.y_result
            t_output = self.t_output
            N = y_result[0].shape[0]//6
            flag = 1
        except Error:
            print("No simulation data!")

        if flag:
            fig = plt.figure()#figsize=(10, 8))
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(212)
            # necessary calculations
            x0, y0, z0, px0, py0, pz0 = np.split(y_result[0], 6)
            x, y, z, px, py, pz = np.split(y_result[-1], 6)
            emittance = []
            for r in y_result:
                x, y, z, px, py, pz = np.split(r, 6)
                emittance.append(np.sqrt(np.linalg.det(np.cov(x, px, bias=1))))
            xm = np.min(x)
            xM = np.max(x)
            # set plot range
            ax1.set(xlabel="x (µm)", ylabel="px (keV/c)", xlim=(xm-10, xM+10), ylim=(-1, 1))#, \
#                     title='Phase Space Distribution')
            ax2.set(xlabel="z (nm)", ylabel="emittance (µm·keV/c)", xlim=(t_output[0], t_output[-1]), \
                    ylim=(emittance[0]-0.01, emittance[-1]+0.01))#, title='Emittance Evolution')
            ax3.set(xlabel="x (µm)", ylabel="z (nm)", xlim=(xm-10, xM+10), ylim=(np.min(z0), np.max(z)))#, \
#                     title='Trajectories')

            phase = ax1.scatter([], [], marker='+')
            emit, = ax2.plot([], [])
            traj = []
            for i in range(N):
                tr, = ax3.plot([], [], 'b-')
                traj.append(tr)

            def init():
                phase.set_offsets(np.array([[], []]))
                emit.set_data([], [])
                for tr in traj:
                    tr.set_data([], [])
                return phase, emit, traj,

            def animate(i):
                x, y, z, px, py, pz = np.split(y_result[i], 6)
                phase.set_offsets(np.vstack((x, px)).transpose())
                emit.set_data(t_output[:i+1], emittance[:i+1])
                for k, tr in enumerate(traj):
                    tr.set_data(y_result[:i+1, k], y_result[:i+1, 2*N+k])
                return phase, emit, traj,

            anim = animation.FuncAnimation(fig, animate, init_func=init, frames=t_output.shape[0], 
                                           interval=20, blit=False)
            anim.save(filename+'.mp4', dpi=150, fps=20)
