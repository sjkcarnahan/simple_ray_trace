'''
Scott Carnahan
simple ray trace - tools and classes to specify and instantiate rays
'''
import numpy as np
from Basilisk.utilities import RigidBodyKinematics as rbk

class Ray:
    def __init__(self):
        self.X = None  # 3 x N position vectors of rays
        self.d = None  # direction vectors of rays in same frame
        return

    def set_pos(self, ray_starts):
        self.X = ray_starts
        return

    def set_dir(self, ray_dirs):
        self.d = ray_dirs
        return

class AngledCircleRayDef:
    # definition/inputs to make a light source which is a set of rays in concentric circles
    # for a less naive generation of concentric circles of rays, vary the number of rays with sqrt(radius) of each ring.
    def __init__(self):
        self.rad = 0.5  # [m] radius of largest circle of rays
        self.angles = [0.]  # [arc sec] angle of rays measure wrt the instrument primary axis. providing a list will generate
                            # multiple sets of rays to be used in multiple runs of the experiment.
        self.num_circ = 15  # number of concentric circles
        self.per_circ = 150  # number of rays per circle

def make_angled_circle_rays(inputs):
    rad_inc = inputs.rad / inputs.num_circ  # radius increment
    theta_inc = np.pi * 2 / inputs.per_circ  # angle increment
    rays_list = []  # set of sets of start points
    rays_d_list = []  # set of sets of directions
    for angle in inputs.angles:
        rays = []
        angle = angle / 3600. * np.pi / 180.  # convert from arc sec to radians
        for i in range(inputs.num_circ):
            r = rad_inc * i
            for j in range(inputs.per_circ):
                # note x = 0 always. We assume the rays start at the y-z plane in the lab frame.
                x, y, z = 0., r * np.cos(theta_inc * j), r * np.sin(theta_inc * j)
                rays.append(np.array([x, y, z]))
        rays = np.array(rays).transpose()
        ray_dirs = np.array([np.array([1, 0, 0])] * np.shape(rays)[1]).transpose()  # rays initialize down x-axis
        DCM = rbk.euler1232C([0., 0., angle]).transpose()
        ray_dirs = np.dot(DCM, ray_dirs)  # rays rotated by given angle
        rays_list.append(rays)
        rays_d_list.append(ray_dirs)
    return rays_list, rays_d_list  # here we have a list of ray sets. one set per angle given. many rays per set

def make_one_edge_ray(rad, angle):
    # rad is radius of primary
    # angle is the desired angle of the ray relative to primary centerline
    # make one ray, starts at the edge of the generating circle at a specified angle. For checking secondary diameter
    x, y, z = 0., rad, 0.,
    L_X = np.array([x,y,z]).reshape([3, 1])
    angle = angle/3600. * np.pi/180.
    dir = np.array([np.cos(angle), -np.sin(angle), 0]).reshape([3, 1])
    return L_X, dir