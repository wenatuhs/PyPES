class Laser(object):
    """ Laser """
    def __init__(self, dist='UC', sigma_r=10, wavelength=266):
        """ Initialize the laser parameters.
        
        Keyword arguments:
        dist -- laser transverse distribution, 'UC' for uniform, 'GC' for gaussian.
        sigma_r -- rms radial size, unit: µm.
        wavelength -- laser wavelength, unit: nm.
        """
        object.__init__(self)
        self.dist = dist
        self.sigma_r = sigma_r
        self.paras = {}
        self.paras['lambda_l'] = wavelength # unit: nm
        self.paras['Ep'] = 2*np.pi*const.c/(wavelength*1e-9)*const.hbar/const.e # unit: eV
    
    def gen_samples(self, N):
        """ Generate N samples that obey the distribution of laser.
        
        Keyword arguments:
        N -- number of samples to be generated.
        
        Returns:
        2*N ndarray -- the coordinates of the generated samples.
        """
        if self.dist == 'UC':
            s = np.random.rand(2, N)
            s[0] = self.sigma_r*np.sqrt(2)*np.sqrt(s[0])
            s[1] *= 2*np.pi
            return np.array([s[0]*np.cos(s[1]), s[0]*np.sin(s[1])])
        elif self.dist == 'GC':
            return self.sigma_r/np.sqrt(2)*np.random.randn(2, N)
        else:
            return None
        
    def plot_samples(self, samples, save=0, orientation='v', ax=None, divider=None):
        """ Plot the samples.

        Keyword arguments:
        samples -- 2*N ndarray, the coordinates of the samples.
        save -- save the plot or not.
        orientation -- choose between 'v' and 'h', specify the colorbar position.
        ax -- if not None, plot the samples on a given ax. 
        divider -- use the given divider to add new colorbar.
        """
        # mark if generate a new figure & ax
        flag = 0
        if ax is None:
            flag = 1
            fig = plt.figure()#figsize=(8, 8))
            ax = fig.add_subplot(111)
        else:
            fig = ax.get_figure()
        if divider is None:
            divider = make_axes_locatable(ax)
#         ax.grid(True)

        if flag:
            ax.set(xlabel='x (µm)', ylabel='y (µm)', aspect='equal', \
                   xlim=(-3*self.sigma_r, 3*self.sigma_r), ylim=(-3*self.sigma_r, 3*self.sigma_r))
        if self.dist == 'UC':
            ax.scatter(samples[0], samples[1], c='b', s=5, marker='.', alpha=0.5, edgecolors='none')
            # text box for comments
            comment = '$\mathrm{Dist:\,Uniform}$\n$r=%.2f\,\mathrm{\mu m}$\n$\lambda=%.2f\,\mathrm{nm}$'\
                      % (self.sigma_r*np.sqrt(2), self.paras['lambda_l'])
            props = dict(facecolor='w', edgecolor='none')
            if flag:
                ax.text(0.05, 0.95, comment, transform=ax.transAxes, verticalalignment='top', bbox=props)
        elif self.dist == 'GC':
            # normalized intensity by the center intensity
            colors = np.exp(-np.sum((samples-np.mean(samples, axis=1).reshape((2, 1)))**2, axis=0)/self.sigma_r**2)
            trans = ax.scatter(samples[0], samples[1], c=colors, s=5, marker='.', alpha=0.5, \
                               cmap=cm.coolwarm, edgecolors='none')
            if orientation == 'v':
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = fig.colorbar(trans, cax=cax)
                cbar.ax.set_ylabel('relative intensity (arb. units.)')
                cbar.set_alpha(1)
                cbar.draw_all()
            elif orientation == 'h':
                cax = divider.append_axes("top", size="5%", pad=0.1)
                cbar = fig.colorbar(trans, cax=cax, orientation='horizontal')
                cbar.ax.xaxis.tick_top()
                cbar.ax.set_xlabel('relative intensity (arb. units.)')
                cbar.ax.xaxis.set_label_position('top')
                cbar.set_alpha(1)
                cbar.draw_all()
            else:
                pass
            # text box for comments
            comment = '$\mathrm{Dist:\,Gaussian}$\n$\sigma_r = %.2f\,\mathrm{\mu m}$\n$\lambda=%.2f\,\mathrm{nm}$'\
                      % (self.sigma_r, self.paras['lambda_l'])
            props = dict(facecolor='w', edgecolor='none')
            if flag:
                ax.text(0.05, 0.95, comment, transform=ax.transAxes, verticalalignment='top', bbox=props)
        else:
            pass
        
        if save:
            fig.savefig('laser.png', dpi=300, bbox_inches='tight')
        if flag:
            plt.show()
