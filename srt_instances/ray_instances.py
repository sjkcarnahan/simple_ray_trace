'''
Scott Carnahan
simple ray trace
This file is a set of useful and reusable srt_instances of rays at their start points (light_sources)
'''

from srt_modules import light_sources as ls
import numpy as np
from srt_instances.instrument_instances import cass_inputs

# basic paraxial rays
# This light source is a set of rays in concentric circles, which is the easiest light source to use to
# observe spherical aberrations and coma.
basic_paraxial_ray_inputs = ls.AngledCircleRayDef()
basic_paraxial_ray_inputs.rad = cass_inputs.d_1 / 2.
basic_paraxial_ray_inputs.num_circ = 15
basic_paraxial_ray_inputs.per_circ = 150
basic_paraxial_ray_inputs.angles = [0.]
basic_paraxial_rays = ls.make_angled_circle_rays(basic_paraxial_ray_inputs)[0]


# off-axis angles to try
# use these to see aberrations of the image if light is not paraxial
various_angles = [0., 1., 3., 5., 10., 30., 60., 120., 240., 600., 1200.]
various_angle_inputs = ls.AngledCircleRayDef()
various_angle_inputs.rad = basic_paraxial_ray_inputs.rad
various_angle_inputs.num_circ = basic_paraxial_ray_inputs.num_circ
various_angle_inputs.per_circ = basic_paraxial_ray_inputs.per_circ
various_angle_inputs.angles = various_angles
angled_ray_list = ls.make_angled_circle_rays(various_angle_inputs)

# make an edge ray to determine secondary diameter
# The idea here is to say, what size secondary do I need to catch a 5-arcminute off-axis ray?
# Rather than trying to solve it analytically, just trace the ray and see where it lands on the secondary (or off it)
angle_to_capture = 5. * 60.  # arc seconds
edge_ray = ls.make_one_edge_ray(cass_inputs.d_1 / 2., angle_to_capture)

# make rays coming from 30. arcseconds off axis
thirty_sec_ray_inputs = ls.AngledCircleRayDef()
thirty_sec_ray_inputs.rad = basic_paraxial_ray_inputs.rad
thirty_sec_ray_inputs.angles = [30.]
thirty_sec_ray_inputs.num_circ = 15
thirty_sec_ray_inputs.per_circ = 150
thirty_sec_rays = ls.make_angled_circle_rays(thirty_sec_ray_inputs)[0]

# make rays coming from 60. arcseconds off axis
five_min_ray_inputs = ls.AngledCircleRayDef()
five_min_ray_inputs.rad = basic_paraxial_ray_inputs.rad
five_min_ray_inputs.angles = [60.]
five_min_ray_inputs.num_circ = 15
five_min_ray_inputs.per_circ = 150
five_min_rays = ls.make_angled_circle_rays(five_min_ray_inputs)[0]

# ray inputs for the grating problem
wavelength_list = np.arange(1200., 2100., 100.)
colors = ['violet', 'indigo', 'blue', 'green', 'yellow', 'orange', 'red', 'pink', 'black',
            'violet', 'indigo', 'blue', 'green', 'yellow', 'orange', 'red', 'pink', 'black',
            'violet', 'indigo', 'blue', 'green', 'yellow', 'orange', 'red', 'pink', 'black']  # for plotting spectra