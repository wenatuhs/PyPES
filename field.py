class Field(object):
    """ Cathode surface field """
    def __init__(self, surface, E0):
        """ Initialize the field.
        
        Just setup the surface and the electric field strength, don't do calculations.

        Keyword arguments:
        surface -- surface object, the surface on which we calculate the field distribution.
        E0 -- electric field strength, unit: MV/m.
        """
        object.__init__(self)
        self.paras = {}
        self.paras['E0'] = E0
        self.surface = surface
        
    def gen_field(self, zrange):
        """ Generating surface electric field & potential map and store them.
        
        Note that the field map is normalized. The real field is the normalized field times E0.

        Keyword arguments:
        zrange -- 1*N ndarray, at which z position to calculate the fields, unit: nm.

        Usage:
        Given a position (x [um], y [um], z [nm]), the normalized electric field would be:
        Ex = self.Ex((x, y, z))
        Ey = self.Ey((x, y, z))
        Ez = self.Ez((x, y, z))
        And the potential would be:
        P = self.P((x, y, z))
        """
        num = len(zrange)
        pb = JavaProgressBar(num)
        pb.animate(0, 'initializing...')
        
        P = []
        Ex = []
        Ey = []
        Ez = []
        Z = zrange
        surface = self.surface

        r = fft2(surface.Z)
        kx = 2 * np.pi * fftfreq(len(surface.Z[0, :]), d=surface.dx).reshape(1, -1)
        ky = 2 * np.pi * fftfreq(len(surface.Z[:, 0]), d=surface.dy).reshape(-1, 1)
        k = np.sqrt(kx**2+ky**2)
        for i, z in enumerate(Z):
            pb.animate(i+1, 'calculating field map for z = {:.2f} nm...'.format(z))
            r_n = r * np.e ** (-k * z * 1e-3)
            r_nx = 1j * kx * 1e-3 * r * np.e ** (-k * z * 1e-3)
            r_ny = 1j * ky * 1e-3 * r * np.e ** (-k * z * 1e-3)
            r_nz = k * 1e-3 * r * np.e ** (-k * z * 1e-3)
            s_n = z - ifft2(r_n).real
            s_nx = ifft2(r_nx).real
            s_ny = ifft2(r_ny).real
            s_nz = -1 - ifft2(r_nz).real
            # pick up the valid electric field
            valid = (s_n >= 0)
            P.append(s_n*valid)
            Ex.append(s_nx*valid)
            Ey.append(s_ny*valid)
            Ez.append(s_nz*valid)
        P = np.array(P)
        Ex = np.array(Ex)
        Ey = np.array(Ey)
        Ez = np.array(Ez)
        pb.animate(num, 'generating interpolation functions...')
        p = RegularGridInterpolator((surface.X[0, :], surface.Y[:, 0], Z), P.transpose())
        fx = RegularGridInterpolator((surface.X[0, :], surface.Y[:, 0], Z), Ex.transpose())
        fy = RegularGridInterpolator((surface.X[0, :], surface.Y[:, 0], Z), Ey.transpose())
        fz = RegularGridInterpolator((surface.X[0, :], surface.Y[:, 0], Z), Ez.transpose())
        self.P, self.Ex, self.Ey, self.Ez = p, fx, fy, fz
        pb.animate(num, 'done.', 1)
        
    def int_field(self):
        """ Integrating the surface electric field.

        Returns:
        Interpolation function Ix, Iy

        Usage:
        Given a position (x [um], y [um]), the integration would be:
        Ix = Ix((x, y))
        Iy = Iy((x, y))
        """
        surface = self.surface
        
        r = fft2(surface.Z)
        kx = 2 * np.pi * fftfreq(len(surface.Z[0, :]), d=surface.dx).reshape(1, -1)
        ky = 2 * np.pi * fftfreq(len(surface.Z[:, 0]), d=surface.dy).reshape(-1, 1)
        k = np.sqrt(kx**2+ky**2)
        k[0, 0] = 1e-9 # avoid divided by zero

        r_nx = -1j * kx / np.sqrt(k) * r
        s_nx = np.array(ifft2(r_nx).real)
        r_ny = -1j * ky / np.sqrt(k) * r
        s_ny = np.array(ifft2(r_ny).real)

        Ix = RegularGridInterpolator((surface.X[0, :], surface.Y[:, 0]), s_nx.transpose())
        Iy = RegularGridInterpolator((surface.X[0, :], surface.Y[:, 0]), s_ny.transpose())

        return Ix, Iy
