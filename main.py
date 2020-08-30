surface = Surface('halfcell-2D.asc', '../../surface/')
laser = Laser('GC', 20)
copper = Material('Copper', path='../../material/')
field = Field(surface, 50)

dowell = Dowell(surface, copper, laser, field)
dowell.setup([80, 60], 0, 50)

pe = PEModel(surface, copper, laser, field)
pe.setup([80, 60], 0, 50)

# Surface morphology
surface.plot_surface([0, 5, 0, 5], projection=1, interpolation=1, interpoints=(100, 100), save=1)
surface.plot_surface([0, 5, 0, 5], projection=0, interpolation=1, interpoints=(100, 100), save=1)

# Surface spectrum
surface.plot_spectrum(threshold=1, save=1)

# Laser samples
laser.plot_samples(laser.gen_samples(50000), save=1)

# Configuration
dowell.plot_configuration(50000, save=1, orientation='-')
pe.plot_configuration(50000, save=1, orientation='+')

# Verify the energy conservation
surface.set_baseplane(0)
stat = surface.stat_spectrum()
rms_x = np.sqrt(np.mean(surface.Z**2))
stat['rms_r']/(surface.dx*surface.dy)/np.sqrt(480*752)*1e3  # µm to nm
