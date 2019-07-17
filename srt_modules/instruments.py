'''
Scott Carnahan
simple ray trace - instrument srt_modules and utilities
'''
import numpy as np
from srt_modules import optical_surfaces as surfs

class CassegrainDefinition:
    def __init__(self):
        self.f_num_1 = 3.  # primary f/#
        self.d_1 = 1.  # primary diameter
        self.f_num_total = 15.  # total system f/#
        self.e = 0.15  # distance from primary vertex to image focal point
        self.primary_x = 3.  # x-position of primary vertex in lab frame
        self.orientation_212 = [-np.pi / 2., 0, 0]  # 2-1-2 euler angle [radians] attitude of the system wrt lab frame
        self.focal_plane_offset = 0.  # move the focal plane +/- this much along the instrument axis off of the focus
        return

class Instrument:
    # An instrument is more or less a list of detectors
    def __init__(self):
        self.surfaces = []
        self.detector = None  # but it is also useful to be able to easily access the detector/image plane
        return

    def set_surfaces(self, surfs):
        self.surfaces = surfs
        self.detector = surfs[-1]
        return

    def set_detector(self, detector):
        # this makes it easy to compare detectors
        self.detector = detector
        self.surfaces[-1] = self.detector

def cassegrain_set_up(inputs):  # inputs are a CassegrainDefinition()
    # this method is an algorithm to initialize the surface shapes and locations for a cassegrain scope as defined by
    # CassegrainDefinition()
    # Although it doesn't follow entirely, the inspiration here comes from Chapter 4 of Geometric Optics by Romano.
    # It is mostly basic geometry once you understand the terms, though.
    f_1 = inputs.f_num_1 * inputs.d_1  # primary mirror focal length
    M = inputs.f_num_total / inputs.f_num_1  # magnification of the system
    alpha = inputs.e / f_1
    beta = (M - alpha) / (M + 1.)
    c = 1. - beta
    d = -beta * f_1  # separation of primary and secondary vertices.
    d_2 = c * inputs.d_1  # diameter of the secondary
    f_2 = (f_1 + inputs.e) / 2.
    secondary_a = f_2 - (f_1 + d)
    secondary_b = np.sqrt(f_2 ** 2 - secondary_a ** 2)

    # establish primary mirror
    primary_position = np.array([inputs.primary_x, 0, 0])  # in the lab frame
    primary_a = 1. / (4. * f_1)
    primary_hole_diam = 0.1
    primary = surfs.ParabolicMirrorWithHole(primary_a, inputs.d_1, primary_hole_diam, inputs.orientation_212, primary_position)

    # establish secondary mirror
    secondary_position = np.array([inputs.primary_x + d + secondary_a, 0., 0])  # lab frame
    secondary = surfs.ConvexHyperbolicMirror(secondary_b, secondary_a, d_2, secondary_position, inputs.orientation_212)

    # the "dead spot" here is a hacky tool to kill rays that hit the back of the secondary
    dead_spot_position = secondary_position - np.array([secondary.max_z, 0., 0.])  # lab frame
    dead_spot = surfs.CircleOfDeath(d_2 / 2., dead_spot_position, inputs.orientation_212)

    # the image plane is something for the rays to intersect and be plotted, or imaged
    image_plane_position = primary_position + np.array([inputs.e + inputs.focal_plane_offset, 0., 0.])  # lab frame
    image_plane = surfs.FlatImagePlane(inputs.d_1, inputs.d_1, image_plane_position, inputs.orientation_212)
    return [dead_spot, primary, secondary, image_plane]

def cassegrain_set_up_spherical_detector(f_num_1, d_1, f_num_tot, e, primary_x, image_plane_offset, image_plane_r,
                                         image_plane_max_r, orientation):
    f_num_1 = 3.
    d_1 = 1.
    f_1 = f_num_1 * d_1
    f_num_tot = 15.
    M = f_num_tot / f_num_1
    e = 0.15
    alpha = e / f_1
    beta = (M - alpha) / (M + 1.)
    c = 1. - beta
    d = -beta * f_1  # separation of primary and secondary
    d_2 = c * d_1
    f_2 = (f_1 + e) / 2.
    secondary_a = f_2 - (f_1 + d)
    secondary_b = np.sqrt(f_2 ** 2 - secondary_a ** 2)
    primary_position = np.array([primary_x, 0, 0])
    primary_a = 1. / (4. * f_1)
    primary_hole_diam = 0.1
    primary = surfs.ParabolicMirrorWithHole(primary_a, d_1, primary_hole_diam, orientation, primary_position)
    secondary_position = np.array([primary_x + d + secondary_a, 0., 0])
    secondary = surfs.ConvexHyperbolicMirror(secondary_b, secondary_a, d_2, secondary_position, orientation)
    dead_spot_position = secondary_position - np.array([secondary.max_z, 0., 0.])
    dead_spot = surfs.CircleOfDeath(d_2 / 2., dead_spot_position, orientation)
    image_plane_position = primary_position + np.array([e+image_plane_offset, 0., 0.])
    image_plane = surfs.SphericalImagePlane(image_plane_r, image_plane_max_r, image_plane_position, orientation)
    return [dead_spot, primary, secondary, image_plane]

