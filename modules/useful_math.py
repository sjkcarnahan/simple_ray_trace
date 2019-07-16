'''
Scott Carnahan
simple ray trace - math
'''

import numpy as np

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