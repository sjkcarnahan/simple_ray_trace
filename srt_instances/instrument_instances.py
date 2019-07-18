'''
Scott Carnahan
simple ray trace - instrument srt_instances
Here we have some useful and reusable instantiations of instrument and surface srt_modules.
'''

from srt_modules import instruments
import numpy as np
from copy import deepcopy

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
cylindrical_detector_inputs = instruments.CylindricalDetectorDefinition()
cylindrical_detector_inputs.radius = 1.0
cylindrical_detector_inputs.height = 1.0
cylindrical_detector_inputs.sweep_angle = np.pi
cylindrical_detector_inputs.base_DCM_SL = grating.DCM_SL
cylindrical_detector_inputs.base_position = grating.L_r_L
cylindrical_detector_inputs.offset_distance = grating_inputs.radius
cylindrical_detector = instruments.cylindrical_detector_setup(cylindrical_detector_inputs)

# Make an instrument with a grating and cylindrical detector
grating_cassegrain = instruments.Instrument()
grating_cassegrain.set_surfaces(deepcopy(cass.surfaces))
grating_cassegrain.surfaces[-1] = grating
grating_cassegrain.surfaces.append(deepcopy(cylindrical_detector))
grating_cassegrain.set_detector(deepcopy(cylindrical_detector))