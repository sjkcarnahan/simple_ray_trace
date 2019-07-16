'''
Scott Carnahan
This file builds up some ray tracing module instances to be used in "experiments". It keeps the experiment files much
cleaner and allows for some reuse, especially in the instrument initialization.
'''

from modules import rayTracing as rt
import numpy as np
from Basilisk.utilities import RigidBodyKinematics as rbk

# instrument design inputs
cass_inputs = rt.CassegrainDefinition()  # organizational tool
cass_inputs.f_num_1 = 3.  # primary f number
cass_inputs.d_1 = 1.  # primary diameter [m]
cass_inputs.f_num_total = 15.  # instrument total f/#
cass_inputs.e = 0.15  # distance secondary focus behind primary vertex [m]
cass_inputs.primary_x = 3.  # position of the primary in the lab frame [m]
cass_inputs.focal_plane_offset = 0.  # put the detector this far off of the secondary focus
cass_inputs.orientation_212 = [-np.pi / 2., 0, 0]  # orientation of the instrument wrt lab frame. not provided to me

# instrument for this project
cass = rt.Instrument()  # you need an instrument to add to an experiment
cass.set_surfaces(rt.cassegrain_set_up(cass_inputs))  # and an instrument is essentially a collection of surfaces

# basic paraxial rays
# This light source is a set of rays in concentric circles, which is the easiest light source to use to
# observe spherical aberrations and coma.
basic_paraxial_ray_inputs = rt.AngledCircleRayDef()
basic_paraxial_ray_inputs.rad = cass_inputs.d_1 / 2.
basic_paraxial_ray_inputs.num_circ = 15
basic_paraxial_ray_inputs.per_circ = 150
basic_paraxial_ray_inputs.angles = [0.]
starts, dirs = rt.make_angled_circle_rays(basic_paraxial_ray_inputs)
basic_paraxial_rays = rt.Ray()
basic_paraxial_rays.set_dir(dirs[0])
basic_paraxial_rays.set_pos(starts[0])

# off-axis angles to try
# use these to see aberrations of the image if light is not paraxial
various_angles = [0., 1., 3., 5., 10., 30., 60., 120., 240., 600., 1200.]
various_angle_inputs = rt.AngledCircleRayDef()
various_angle_inputs.rad = basic_paraxial_ray_inputs.rad
various_angle_inputs.num_circ = basic_paraxial_ray_inputs.num_circ
various_angle_inputs.per_circ = basic_paraxial_ray_inputs.per_circ
various_angle_inputs.angles = various_angles
angled_starts, angled_dirs = rt.make_angled_circle_rays(various_angle_inputs)
angled_ray_list = []
for i in range(len(angled_starts)):
    a_ray = rt.Ray()
    a_ray.set_dir(angled_dirs[i])
    a_ray.set_pos(angled_starts[i])
    angled_ray_list.append(a_ray)

# make an edge ray to determine secondary diameter
# The idea here is to say, what size secondary do I need to catch a 5-arcminute off-axis ray?
# Rather than trying to solve it analytically, just trace the ray and see where it lands on the secondary (or off it)
angle_to_capture = 5. * 60.  # arc seconds
edge_ray_X, edge_ray_d = rt.make_one_edge_ray(cass_inputs.d_1 / 2., angle_to_capture)
edge_ray = rt.Ray()
edge_ray.set_pos(edge_ray_X)
edge_ray.set_dir(edge_ray_d)

# make rays coming from 30. arcseconds off axis
thirty_sec_ray_inputs = rt.AngledCircleRayDef()
thirty_sec_ray_inputs.rad = basic_paraxial_ray_inputs.rad
thirty_sec_ray_inputs.angles = [30.]
thirty_sec_ray_inputs.num_circ = 15
thirty_sec_ray_inputs.per_circ = 150
starts, dirs = rt.make_angled_circle_rays(thirty_sec_ray_inputs)
thirty_sec_rays = rt.Ray()
thirty_sec_rays.set_pos(starts[0])
thirty_sec_rays.set_dir(dirs[0])

# make rays coming from 60. arcseconds off axis
five_min_ray_inputs = rt.AngledCircleRayDef()
five_min_ray_inputs.rad = basic_paraxial_ray_inputs.rad
five_min_ray_inputs.angles = [60.]
five_min_ray_inputs.num_circ = 15
five_min_ray_inputs.per_circ = 150
starts, dirs = rt.make_angled_circle_rays(five_min_ray_inputs)
five_min_rays = rt.Ray()
five_min_rays.set_pos(starts[0])
five_min_rays.set_dir(dirs[0])

# grating stuff
surfaces = cass.surfaces[1:]
line_density = 3600.  # per mm
mode = 1
lam = 1600E-10
d = 1. / (line_density * 1000.)
alpha = np.arcsin(mode * lam / d)
rotation_for_beta_zero = rbk.euler2122C([alpha, 0., 0.])
DCM_basic = rbk.euler2122C([-np.pi/2., 0., 0.])
DCM_SL = np.dot(rotation_for_beta_zero, DCM_basic)
grating = rt.RowlandCircle()
grating.set_radius(1.0)
grating.set_line_density(line_density)
f_num_2 = cass_inputs.f_num_total - cass_inputs.f_num_1
offset = grating.r * np.cos(alpha)
focus = surfaces[-1].L_r_L
pos = focus + np.array([offset, 0, 0]).reshape([3, 1])
grating.set_position(pos)
grating.set_DCM(DCM_SL)
grating.set_width(0.24)
grating.set_order(0)
grating.set_wavelength(1600.)

cylindrical_detector = rt.CylindricalDetector()
cylindrical_detector.set_radius(1.0)
cylindrical_detector.set_height(1.0)
cylindrical_detector.set_sweep_angle(np.pi)
dcm_cyl = np.dot(rbk.euler2122C([np.pi, 0., 0.]), DCM_SL)
dcm_rot = np.dot(rbk.euler1232C([0., 0., -np.pi/2.]), dcm_cyl)
cylindrical_detector.set_DCM(dcm_rot)
offset = np.dot(DCM_SL.transpose() , np.array([0., 0., 1.0]))
cylindrical_detector.set_position(pos + offset.reshape([3, 1]))
cylindrical_detector.set_y_limits()

