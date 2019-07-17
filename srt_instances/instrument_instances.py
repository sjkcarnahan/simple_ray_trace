'''
Scott Carnahan
simple ray trace - instrument srt_instances
Here we have some useful and reusable instantiations of instrument and surface srt_modules.
'''

from srt_modules import instruments, light_sources as ls, optical_surfaces as surfs
import numpy as np
from copy import deepcopy
from srt_modules.useful_math import euler2122C, euler1232C

# instrument design inputs
cass_inputs = instruments.CassegrainDefinition()  # organizational tool
cass_inputs.f_num_1 = 3.  # primary f number
cass_inputs.d_1 = 1.  # primary diameter [m]
cass_inputs.f_num_total = 15.  # instrument total f/#
cass_inputs.e = 0.15  # distance secondary focus behind primary vertex [m]
cass_inputs.primary_x = 3.  # position of the primary in the lab frame [m]
cass_inputs.focal_plane_offset = 0.  # put the detector this far off of the secondary focus
cass_inputs.orientation_212 = [-np.pi / 2., 0, 0]  # orientation of the instrument wrt lab frame. not provided to me

# instrument for this project
cass = instruments.Instrument()  # you need an instrument to add to an experiment
cass.set_surfaces(instruments.cassegrain_set_up(cass_inputs))  # and an instrument is essentially a collection of surfaces

# grating stuff
# inputs to and creation of rowland circle
grating_inputs = instruments.RowlandCircleDefinition()
grating_inputs.line_density = 3600.
grating_inputs.mode = 1  # mode/order. design the placement of the detector and grating around 1st order diffraction
grating_inputs.lam = 1600E-10  # design wavelength
grating_inputs.e212 = [-np.pi/2., 0., 0.]
grating_inputs.radius = 1.0
grating_inputs.focus = cass.surfaces[-1].L_r_L
grating_inputs.order = 0
grating_inputs.width = 0.24
grating = instruments.rowland_circle_setup(grating_inputs)


# a cylindrical detector will be used to read the results of the grating expriment
cylindrical_detector = surfs.CylindricalDetector()
cylindrical_detector.set_radius(1.0)
cylindrical_detector.set_height(1.0)
cylindrical_detector.set_sweep_angle(np.pi)
dcm_cyl = np.dot(euler2122C([np.pi, 0., 0.]), grating.DCM_SL)
dcm_rot = np.dot(euler1232C([0., 0., -np.pi/2.]), dcm_cyl)
cylindrical_detector.set_DCM(dcm_rot)
offset = np.dot(grating.DCM_SL.transpose() , np.array([0., 0., 1.0]))
cylindrical_detector.set_position(grating.L_r_L + offset.reshape([3, 1]))
cylindrical_detector.set_y_limits()

# Make an instrument with a grating
grating_cassegrain = instruments.Instrument()
grating_cassegrain.set_surfaces(deepcopy(cass.surfaces))
grating_cassegrain.surfaces[-1] = grating
grating_cassegrain.surfaces.append(cylindrical_detector)
grating_cassegrain.set_detector(cylindrical_detector)