from modules import rayTracing as rt
import numpy as np

# instrument design inputs
# These inputs were provides by Prof Jim Green of CU Boulder for the 1st class project in ASTR 5760 Spring 2019
cass_inputs = rt.CassegrainDefinition()  # organizational tool
cass_inputs.f_num_1 = 3.  # primary f number
cass_inputs.d_1 = 1.  # primary diameter [m]
cass_inputs.f_num_total = 15.  # instrument total f/#
cass_inputs.e = 0.15  # distance secondary focus behind primary vertex [m]
cass_inputs.primary_x = 3.  # position of the primary in the lab frame [m]
cass_inputs.focal_plane_offset = 0.  # put the detector this far off of the secondary focus
cass_inputs.orientation_212 = [-np.pi / 2., 0, 0]  # orientation of the instrument wrt lab frame. part of my formulation

# instrument for this project
cass = rt.Instrument()  # you need an instrument to add to an experiment
cass.set_surfaces(rt.cassegrain_set_up(cass_inputs))  # and an instrument is essentially a collection of surfaces

# basic paraxial rays
basic_paraxial_ray_inputs = rt.AngledCircleRayDef()
basic_paraxial_ray_inputs.rad = cass_inputs.d_1 / 2.
basic_paraxial_ray_inputs.num_circ = 15
basic_paraxial_ray_inputs.per_circ = 150
basic_paraxial_ray_inputs.angles = [0.]
starts, dirs = rt.make_angled_circle_rays(basic_paraxial_ray_inputs)
basical_paraxial_rays = rt.Ray()
basical_paraxial_rays.set_dir(dirs[0])
basical_paraxial_rays.set_pos(starts[0])

# off-axis angles to try
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
angle_to_capture = 5. * 60.
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

