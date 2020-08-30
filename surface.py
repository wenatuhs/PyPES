class Surface(object):
    """ Surface topography """
    def __init__(self, name, path=''):
        """ Initialize a surface.

        Keyword arguments:
        name -- filename of the surface data file, with extension.
        path -- path to the surface data file, default value is ''.
        """
        object.__init__(self)        
        try:
            filename = os.path.join(path, name)
            f = open(filename)
            count = 0
            info = {"x-pixels": None,
                    "y-pixels": None,
                    "x-length": None,
                    "y-length": None}
            Z = []
            
            for line in f:
                count += 1
                if line.startswith('#'):        
                    words = line.split()
                    if words[1] in info.keys():
                        info[words[1]] = int(words[3])
                elif count < 14+info["y-pixels"]:
                    Z.append([float(s) for s in line.split()])
                else:
                    break
            f.close()
            
            X = np.linspace(0.0, info["x-length"]*1e-3, info["x-pixels"]) # unit: µm
            Y = np.linspace(0.0, info["y-length"]*1e-3, info["y-pixels"]) # unit: µm
            
            self.dx = info['x-length']*1e-3/float(info['x-pixels']-1) # unit: µm
            self.dy = info['y-length']*1e-3/float(info['y-pixels']-1) # unit: µm
            self.xm = info['x-length']*1e-3 # unit: µm
            self.ym = info['y-length']*1e-3 # unit: µm
            self.info = info
            self.X, self.Y = np.meshgrid(X, Y)
            self.Z = np.array(Z) # unit: nm
            self.surf = RegularGridInterpolator((self.X[0, :], self.Y[:, 0]), 
                                                self.Z.transpose()) # surface interp function
        except IOError:
            print("Sorry, IOError!")
        
    def partial(self):
        """ Get the partial interpolation function of the surface.
        
        Using Fourier transform to perform the partial calculation.
        Note that the unit of the partial is nm/µm.
        
        Returns:
        px -- partial R / partial x interp func.
        py -- partial R / partial y interp func.
        """
        r = fft2(self.Z)
        kx = 2 * np.pi * fftfreq(len(self.Z[0, :]), d=self.dx).reshape(1, -1)
        ky = 2 * np.pi * fftfreq(len(self.Z[:, 0]), d=self.dy).reshape(-1, 1)

        r_nx = 1j * kx * r
        s_nx = ifft2(r_nx).real
        s_nx = np.array(s_nx)
        r_ny = 1j * ky * r
        s_ny = ifft2(r_ny).real
        s_ny = np.array(s_ny)

        px = RegularGridInterpolator((self.X[0, :], self.Y[:, 0]), s_nx.transpose())
        py = RegularGridInterpolator((self.X[0, :], self.Y[:, 0]), s_ny.transpose())

        return px, py
        
    def plot_surface(self, area=None, projection=0, interpolation=0, interpoints=(0, 0), aspect=None, 
                     save=0, orientation='v', ax=None):
        """ Draw the surface.

        Keyword arguments:
        area -- the plotting area, format is [x0, x1, y0, y1], unit is µm.
            Default is None, which means the whole surface area.
        projection -- make the projection surface plot or not (3D surface plot), choose between 0 and 1.
        interpolation -- whether perform the interpolation on the raw surface data, choose between 0 and 1.
        interpoints -- if interpolation is set to 1, then interpoints (m, n) means interpolate m points for x and
            n points for y.
        aspect -- the aspect of the frame. Only has effect when projection is set to 1.
            To set the frame aspect to be equal, set aspect to 'equal'.
        save -- save the plot or not.
        orientation -- choose between 'v' and 'h', specify the colorbar position.
        ax -- if ax is not None, plot the samples on a given ax.
            
        Returns:
        divider -- if ax is given, for future use.
        """
        flag = 0
        if ax is None:
            flag = 1
            fig = plt.figure()#figsize=(12, 8))
        else:
            fig = ax.get_figure()
        
        if not area:
            area = (0, self.xm, 0, self.ym)
            
        slicex = slice(int(area[0]/self.dx), int(area[1]/self.dx)+2)
        slicey = slice(int(area[2]/self.dy), int(area[3]/self.dy)+2)
        X, Y, Z = self.X[slicey, slicex], self.Y[slicey, slicex], self.Z[slicey, slicex]
        
        if interpolation:                        
            rbf = Rbf(X, Y, Z, epsilon=2)
            X = np.linspace(area[0], area[1], interpoints[0])
            Y = np.linspace(area[2], area[3], interpoints[1])
            X, Y = np.meshgrid(X, Y)
            Z = rbf(X, Y)
        
        if not projection:
            if flag:
                ax = fig.add_subplot(111, projection='3d')
            divider = make_axes_locatable(ax)
            ax.set(xlabel="x (µm)", ylabel="y (µm)", zlabel="z (nm)",\
                   xlim=(area[0], area[1]), ylim=(area[2], area[3]), 
                   aspect=0.9)#, title="Surface Topography")
#             for item in ([ax.xaxis.label, ax.yaxis.label, ax.zaxis.label] + \
#                          ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels()):
#                 item.set_fontsize(18)
            
            surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=cm.afmhot, linewidth=0.05, alpha=1.0)
#             ax.zaxis.set_major_locator(LinearLocator(10))
#             ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        else:
            if flag:
                ax = fig.add_subplot(111)
            divider = make_axes_locatable(ax)
            if aspect == 'equal':
                ax.set(aspect=aspect)
            ax.set(xlabel="x (µm)", ylabel="y (µm)",\
                   xlim=(area[0], area[1]), ylim=(area[2], area[3]))#, title="Surface Topography")
            surf = ax.pcolormesh(X, Y, Z, cmap=cm.afmhot)
            
            if orientation == 'v':
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = fig.colorbar(surf, cax=cax)
                cbar.ax.set_ylabel('z (nm)')
            elif orientation == 'h':
                cax = divider.append_axes("top", size="5%", pad=0.1)
                cbar = fig.colorbar(surf, cax=cax, orientation='horizontal')
                cbar.ax.xaxis.tick_top()
                cbar.ax.set_xlabel('z (nm)')
                cbar.ax.xaxis.set_label_position('top')
            else:
                pass

        fig.tight_layout()
        if save:
            fig.savefig('surface.png', dpi=300, bbox_inches='tight')
        if flag:
            plt.show()
        else:
            return divider
        
    def plot_spectrum(self, area=None, threshold=1, save=0):
        """ Draw the spectrum of the surface.

        Keyword arguments:
        area -- the plotting area, format is [x0, x1, y0, y1], unit is µm.
            Default is None, which means the whole surface area.
        threshold -- the saturation threshold of the spectrum. Value in the spectrum which is larger
            then threshold will be set to threshold.
        save -- save the plot or not.
        """
        if not area:
            area = (0, self.xm, 0, self.ym)
            
        slicex = slice(int(area[0]/self.dx), int(area[1]/self.dx)+2)
        slicey = slice(int(area[2]/self.dy), int(area[3]/self.dy)+2)
        Z = self.Z[slicey, slicex]
        
        R = fftshift(np.abs(fft2(Z)))*self.dx*self.dy*1e-3 # change unit to µm³
        R[R >= threshold] = threshold
        kx = 2 * np.pi * fftshift(fftfreq(len(Z[0, :]), d=self.dx))
        ky = 2 * np.pi * fftshift(fftfreq(len(Z[:, 0]), d=self.dy))
        Kx, Ky = np.meshgrid(kx, ky)
        
        fig, ax = plt.subplots(1, 1)#, figsize=(12, 8))
        ax.set(xlim=(kx[0], kx[-1]), ylim=(ky[0], ky[-1]), aspect='equal', 
               xlabel="kx (1/µm)", ylabel="ky (1/µm)")
        surf = ax.pcolormesh(Kx, Ky, R, cmap=cm.afmhot)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(surf, cax=cax)
        cbar.ax.set_ylabel('amplitude (µm³)')
#         for item in ([ax.xaxis.label, ax.yaxis.label, cbar.ax.yaxis.label] + \
#                      ax.get_xticklabels() + ax.get_yticklabels() + cbar.ax.get_yticklabels()):
#             item.set_fontsize(18)
        # change one of the cbar labels
        cbar.draw_all()
        labels = [item.get_text() for item in cbar.ax.get_yticklabels()]
        try:
            digits = len(labels[0].split('.')[1])
        except:
            digits = 0
        labels[-1] = '≥ {{:.{}f}}'.format(digits).format(threshold)
        cbar.ax.set_yticklabels(labels)
        
        fig.tight_layout()
        if save:
            fig.savefig('spectrum.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def set_baseplane(self, base=0):
        """ Set the baseplane of the surface.
    
        Set mean(Z) to the given base, also update the surf() method.

        Keyword arguments:
        base -- the base plane position, unit: nm. [0]
        """
        old_base = np.mean(self.Z) # nm
        self.Z += (base-old_base)
        self.surf = RegularGridInterpolator((self.X[0, :], self.Y[:, 0]), 
                                            self.Z.transpose())
        
    def stat_spectrum(self, area=None):
        """ Statistic the spectrum of the surface.

        Keyword arguments:
        area -- the stat area, format is [x0, x1, y0, y1], unit is µm.
            Default is None, which means the whole surface area.
        kind -- kind of statistic, 'rms' or 'std'. ['rms']
        
        Returns:
        stat_spec -- dictionary of the statistic results.            
            * 'rms_kx': rms of kx
            * 'rms_ky': rms of ky
            * 'rms_r': rms of r (in frequency domain)
        """
        # Define temporal rms method
        rms = lambda a, axis=None, weights=None: np.sqrt(np.average(a**2, axis, weights))
        
        if not area:
            area = (0, self.xm, 0, self.ym)
            
        slicex = slice(int(area[0]/self.dx), int(area[1]/self.dx)+2)
        slicey = slice(int(area[2]/self.dy), int(area[3]/self.dy)+2)
        Z = self.Z[slicey, slicex]
        
        R = fftshift(np.abs(fft2(Z)))*self.dx*self.dy*1e-3 # change unit to µm³
        kx = 2 * np.pi * fftshift(fftfreq(len(Z[0, :]), d=self.dx))
        ky = 2 * np.pi * fftshift(fftfreq(len(Z[:, 0]), d=self.dy))
        Kx, Ky = np.meshgrid(kx, ky)
        
        return {'rms_kx': rms(Kx, weights=R), 'rms_ky': rms(Ky, weights=R), 'rms_r': rms(R)}
