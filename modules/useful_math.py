'''
Scott Carnahan
simple ray trace - math
'''

import numpy as np
import math

def solve_quadratic(a, b, c, pm):
    # standard form of the quadratic equation
    if pm == 1:
        return (-b + np.sqrt(b**2. - 4. * a * c)) / 2. / a
    elif pm == -1:
        return (-b - np.sqrt(b**2. - 4. * a * c)) / 2. / a

def mullers_quadratic_equation(a, b, c, pm):
    # an alternative form of the quadratic equation that doesn't
    # fail if a == 0.
    # useful if rays are coming straight in at a parabolic mirror, specifically
    if pm == 1:
        return (2 * c) / (-b + np.sqrt(b**2 - 4. * a * c))
    elif pm == -1:
        return (2 * c) / (-b - np.sqrt(b**2 - 4. * a * c))

def rms_image(image_plane, L_X):
    pts = image_plane.extract_image(L_X)
    xs = pts[0, :]
    ys = pts[1, :]
    xs, ys = xs[~np.isnan(xs) & ~np.isnan(ys)], ys[~np.isnan(xs) & ~np.isnan(ys)]
    av_x, av_y = np.average(xs), np.average(ys)
    av_pt = np.array([av_x, av_y]).reshape([2, 1])
    pts = np.vstack([xs, ys])
    diff = pts - av_pt
    l2norm_square = (diff*diff).sum(axis=0)
    rms = np.sqrt(np.average(l2norm_square))
    return rms

'''
the euler angle to DCM functions below are taken directly from the Basilisk code base
http://hanspeterschaub.info/bskHtml/index.html
The math is explained in Analytical Mechanics of Space Systems by Schaub
'''
def euler1232C(q):
    """
    euler1232C

    	C = euler1232C(Q) returns the direction cosine
    	matrix in terms of the 1-2-3 euler angles.
    	Input Q must be a 3x1 vector of euler angles.
    """

    st1 = math.sin(q[0])
    ct1 = math.cos(q[0])
    st2 = math.sin(q[1])
    ct2 = math.cos(q[1])
    st3 = math.sin(q[2])
    ct3 = math.cos(q[2])

    C = np.identity(3)
    C[0, 0] = ct2 * ct3
    C[0, 1] = ct3 * st1 * st2 + ct1 * st3
    C[0, 2] = st1 * st3 - ct1 * ct3 * st2
    C[1, 0] = -ct2 * st3
    C[1, 1] = ct1 * ct3 - st1 * st2 * st3
    C[1, 2] = ct3 * st1 + ct1 * st2 * st3
    C[2, 0] = st2
    C[2, 1] = -ct2 * st1
    C[2, 2] = ct1 * ct2

    return C

def euler2122C(q):
    """
    euler2122C

    	C = euler2122C(Q) returns the direction cosine
    	matrix in terms of the 2-1-2 euler angles.
    	Input Q must be a 3x1 vector of euler angles.
    """

    st1 = math.sin(q[0])
    ct1 = math.cos(q[0])
    st2 = math.sin(q[1])
    ct2 = math.cos(q[1])
    st3 = math.sin(q[2])
    ct3 = math.cos(q[2])

    C = np.identity(3)
    C[0, 0] = ct1 * ct3 - ct2 * st1 * st3
    C[0, 1] = st2 * st3
    C[0, 2] = -ct3 * st1 - ct1 * ct2 * st3
    C[1, 0] = st1 * st2
    C[1, 1] = ct2
    C[1, 2] = ct1 * st2
    C[2, 0] = ct2 * ct3 * st1 + ct1 * st3
    C[2, 1] = -ct3 * st2
    C[2, 2] = ct1 * ct2 * ct3 - st1 * st3

    return C