'''
Scott Carnahan
Experiment - Spherical Detector Design/Placement
Spring 2019
Again, this experiment is similar to the find_image_plane_5 experiment, except there is an addition search dimension for
the radius of the detector. The results should indicate a continuum of pairs of radii and positions that minimize the
rms of the points on the detector. Note also that the SphericalDetector() image coordinates are in RA and DEC rather
than being projected to a flat x, y.
'''

import numpy as np
import matplotlib.pyplot as plt
from srt_modules import experiment, optical_surfaces as surfs
from srt_instances import instrument_instances as ii, ray_instances as ri

# plotting flags
show_plots = True
save_plots = True
make_plots = show_plots or save_plots

# create an experiment
exp_sph = experiment.Experiment()

# set up the instrument
exp_sph.add_instrument(ii.cass)

# choose some rays
rays = ri.five_min_rays
exp_sph.set_ray_starts(rays.X)
exp_sph.set_ray_start_dir(rays.d)

# make a spherical detector
spherical_detector = surfs.SphericalDetector()
spherical_detector.L_r_L = np.zeros(3).reshape([3,1])
spherical_detector.DCM_SL = exp_sph.instrument.detector.DCM_SL
spherical_detector.w = .5
exp_sph.instrument.set_detector(spherical_detector)

# I messed around a bit to find good guess to check within
rad_min = .9999 * spherical_detector.r
rad_max = 1.0001 * spherical_detector.r
rad_space = np.linspace(2.624, 2.626, 10)
pos_space = np.linspace(.5245, .5255, 10)

# this is just to be able to make a good surface plot instead of a scatter
R, P = np.meshgrid(rad_space, pos_space)
rms_out = np.zeros(np.shape(R))

# loop over all the combinations of curvature and position
for p in range(len(pos_space)):
    for r in range(len(rad_space)):
        spherical_detector.r = R[p][r]
        spherical_detector.L_r_L[0][0] = P[p][r]
        exp_sph.reset()
        rms_out[p][r] = exp_sph.run().rms

# plot it
fig = plt.figure()
ax = fig.add_subplot('111', projection='3d')
ax.plot_surface(R, P, rms_out)
ax.set_title('Spherical Detector RMS optimization')
ax.set_xlabel('Detector Radius [m]')
ax.set_ylabel('Detector Position [m]')
ax.set_zlabel('log RMS of image [um]')
ax.set_zscale('log')
plt.show()
plt.close('all')




