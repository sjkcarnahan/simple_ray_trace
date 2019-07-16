'''
Scott Carnahan
A ray tracing library I wrote for 5760
Spring 2019
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Basilisk.utilities import RigidBodyKinematics as rbk

def quadratic_eqn(t, *args):
    # given the independent variable, and args=(a,b,c)
    # returns at2 + bt + c
    # used for numerical solution of quadratic equation
    # don't use this anymore.
    return args[0] * t**2 + args[1] * t + args[2]

def solve_quadratic(a, b, c, pm):
    # standard fo rm of the quadratic equation
    if pm == 1:
        return (-b + np.sqrt(b**2. - 4. * a * c)) / 2. / a
    elif pm == -1:
        return (-b - np.sqrt(b**2. - 4. * a * c)) / 2. / a

def mullers_quadratic_equation(a, b, c, pm):
    # an alternative form of the quadratic equation that doesn't
    # fail if a == 0.
    # useful if rays are coming straight in at a parabolic mirror
    if pm == 1:
        return (2 * c) / (-b + np.sqrt(b**2 - 4. * a * c))
    elif pm == -1:
        return (2 * c) / (-b - np.sqrt(b**2 - 4. * a * c))

class parabolicMirrorWithHole:
    # a symmetric paraboloid
    # a is the scaling factor. a(x2 + y2) = z
    # outer diam is the mirror size
    #inner diam allows for a centered hole in the mirror
    # S is the surface frame
    # L is the lab frame
    # e212 is the euler 2-1-2 rotation describing the orientation of the surface wrt the lab frame.
    # L_r_L is the lab-frame-resolved position of the vertex of the paraboloid
    def __init__(self, a, outer_diam, inner_diam, e212, L_r_L):
        self.a = a # slope
        self.outer_diam = outer_diam
        self.inner_diam = inner_diam
        self.max_z = 10.
        self.min_z = 0.
        self.set_limits()
        self.DCM_SL = rbk.euler2122C(e212)
        self.L_r_L = L_r_L.reshape([3, 1])
        self.S_focus = np.array([0., 0., 1 / 4. / a])
        self.L_focus = np.dot(self.DCM_SL.transpose(), np.array([0., 0., 1 / 4. / a])) + self.L_r_L.reshape([3, ])
        self.name = 'primary'

    def equation(self, xs, ys):
        # defining equation for a paraboloid
        # useful for meshing the surface to plot
        return self.a * (xs **2 + ys**2)

    def focus(self):
        # focus at 1/(4a)
        f = np.array([0., 0., 1. / 4. / self.a])
        return np.dot(self.DCM_SL.transpose(), f)

    def normal(self, L_X):
        # given points on the paraboloid, gives unit normal vectors.
        S_X = np.dot(self.DCM_SL, L_X - self.L_r_L)
        xs = S_X[0, :]
        ys = S_X[1, :]
        num = np.shape(ys)
        S_N = np.array([2 * self.a * xs, 2 * self.a * ys, -np.ones(num)])
        S_N_hat = S_N / np.linalg.norm(S_N, axis=0)
        L_N_hat = np.dot(self.DCM_SL.transpose(), S_N_hat)
        return L_N_hat

    def set_limits(self):
        # because everything is done in the surface frame
        # I can rule out hits by their z-value
        # this sets the mirror max/min z
        rad = self.outer_diam / 2.
        self.max_z = self.a * rad ** 2
        rad = self.inner_diam / 2.
        self.min_z = self.a * rad ** 2
        return

    def surface_points(self, n=50):
        # gives X, Y, Z points for the surface to be scatter-plotted
        xs = np.linspace(-self.outer_diam/2, self.outer_diam/2, n).tolist()
        x_out = []
        ys = np.linspace(-self.outer_diam / 2, self.outer_diam /2, n).tolist()
        y_out = []
        z_out = []
        for x in xs:
            for y in ys:
                x_out.append(x)
                y_out.append(y)
                z_out.append(self.equation(x, y))
        x_out = np.array(x_out)
        y_out = np.array(y_out)
        z_out = np.array(z_out)
        x_out = x_out[(z_out <= self.max_z) & (z_out >= self.min_z)]
        y_out = y_out[(z_out <= self.max_z) & (z_out >= self.min_z)]
        z_out = z_out[(z_out <= self.max_z) & (z_out >= self.min_z)]
        X = np.vstack([x_out, y_out, z_out])
        X = np.dot(self.DCM_SL.transpose(), X)
        X = X + self.L_r_L
        return X

    def surface_mesh(self):
        # give x, y, z points for the surface to be surface plotted
        out_rad = self.outer_diam / 2.
        in_rad = self.inner_diam / 2.
        rad_points = np.linspace(in_rad, out_rad, 2)
        theta_points = np.linspace(0., np.pi * 2., 20)
        R, T = np.meshgrid(rad_points, theta_points)
        X, Y = R * np.cos(T), R * np.sin(T)
        Z = self.equation(X, Y)
        for i in range(np.shape(Z)[0]): # transpose into lab frame
            for j in range(np.shape(Z)[1]):
                vec = np.array([X[i,j], Y[i,j], Z[i,j]])
                vec = np.dot(self.DCM_SL.transpose(), vec)
                vec = vec + self.L_r_L.reshape(3,)
                X[i,j], Y[i,j], Z[i, j] = vec[0], vec[1], vec[2]
        return X, Y, Z

    def intersect_rays(self, L_X_0, L_X_d):
        # takes in ray starts and direction unit vectors in lab frame
        num = np.shape(L_X_0)[1]
        S_X_0 = np.dot(self.DCM_SL, L_X_0 - self.L_r_L)
        S_X_d = np.dot(self.DCM_SL, L_X_d)
        x0s = S_X_0[0, :]
        xds = S_X_d[0, :]
        y0s = S_X_0[1, :]
        yds = S_X_d[1, :]
        z0s = S_X_0[2, :]
        zds = S_X_d[2, :]
        A = yds ** 2 + xds ** 2
        B = 2 * (x0s * xds + y0s * yds) - zds / self.a
        C = x0s**2 + y0s**2 -z0s / self.a
        non_nan = ~np.isnan(x0s)
        ts = mullers_quadratic_equation(A[non_nan], B[non_nan], C[non_nan], -1)
        S_X_1 = S_X_0
        S_X_1[:, non_nan] = S_X_0[:, non_nan] + ts * S_X_d[:, non_nan]
        L_X_1 = np.dot(self.DCM_SL.transpose(), S_X_1) + self.L_r_L
        return L_X_1

    def miss_rays(self, L_X):
        # L_X is an intersection point for a parabolloid that is infinite with no holes
        S_X = np.dot(self.DCM_SL, L_X - self.L_r_L)
        zs = S_X[2, :]
        zs[np.isnan(zs)] = -1.
        temp = S_X.transpose()
        temp[(zs > self.max_z) | (zs < self.min_z)] = np.array([np.nan] * 3)
        S_X = temp.transpose()
        L_X = np.dot(self.DCM_SL.transpose(), S_X) + self.L_r_L
        return L_X

    def reflect_rays(self, L_X, L_d_i):
        # takes an intersect point L_X and incoming direction L_d_i
        # produces an outgoing direction L_d_o
        # is there a better vectorized way of doing this?
        num = np.shape(L_X)[1]
        L_N_hat = self.normal(L_X)
        L_d_o = []
        for i in range(num):
            incoming = L_d_i[:,i]
            nHat = L_N_hat[:,i]
            M = np.eye(3) -2 * np.outer(nHat, nHat)
            L_d_o.append(np.dot(M, incoming))
        return np.array(L_d_o).transpose()  # just put it back into 3xN array format

class circleOfDeath:
    # a circular surface that kills rays (back of a mirror perhaps)
    # defined in the x-y frame with a displacement and rotation
    def __init__(self, r, L_r_L, e212):
        self.r = r
        self.L_r_L = L_r_L.reshape([3, 1])
        self.DCM_SL = rbk.euler2122C(e212)
        self.name = "dead_spot"

    def miss_rays(self, L_X_0):
        return L_X_0

    def surface_points(self, num=10):
        rs = np.linspace(0, self.r, num)
        ts = np.linspace(0, 2 * np.pi, num)
        L_points = []
        for r in rs:
            for t in ts:
                x = r * np.cos(t)
                y = r * np.sin(t)
                if x**2 + y**2 > self.r:
                    L_points.append(np.array([np.nan] * 3))
                else:
                    S_point = np.array([x, y, 0])
                    L_points.append(np.dot(self.DCM_SL.transpose(), S_point) + self.L_r_L.reshape([3, ]))
        return np.array(L_points).transpose()

    def surface_mesh(self):
        out_rad = self.r
        rad_points = np.linspace(0., out_rad, 2)
        theta_points = np.linspace(0., np.pi * 2., 30)
        R, T = np.meshgrid(rad_points, theta_points)
        X, Y = R * np.cos(T), R * np.sin(T)
        Z = np.zeros(np.shape(X))
        for i in range(np.shape(Z)[0]):
            for j in range(np.shape(Z)[1]):
                vec = np.array([X[i,j], Y[i,j], Z[i,j]])
                vec = np.dot(self.DCM_SL.transpose(), vec)
                vec = vec + self.L_r_L.reshape(3,)
                X[i,j], Y[i,j], Z[i, j] = vec[0], vec[1], vec[2]
        return X, Y, Z

    def reflect_rays(self, L_X, L_d):
        return L_d

    def intersect_rays(self, L_X_0, L_X_d):
        # takes in ray starts and direction unit vectors in lab frame
        num = np.shape(L_X_0)[1]
        S_X_0 = np.dot(self.DCM_SL, L_X_0 - self.L_r_L)
        S_X_d = np.dot(self.DCM_SL, L_X_d)
        x0s = S_X_0[0, :]
        xds = S_X_d[0, :]
        y0s = S_X_0[1, :]
        yds = S_X_d[1, :]
        z0s = S_X_0[2, :]
        zds = S_X_d[2, :]
        ts = -z0s / zds
        xs = x0s + ts * xds
        ys = y0s + ts * yds
        S_X_1 = S_X_0 + ts * S_X_d
        temp = S_X_1.transpose()
        temp[xs**2. + ys**2. <= self.r ** 2] = np.array([np.nan] * 3)
        S_X_1 = temp.transpose()
        L_X_1 = np.dot(self.DCM_SL.transpose(), S_X_1) + self.L_r_L
        return L_X_1

class ConvexHyperbolicMirror:
    def __init__(self, b, a, diam, L_r_L, e212):
        # (x2 + y2)/b2 - z2/a2 = -1 and take the positive solution
        # where b**2 = f**2 - a**2. and f is the focal distance.
        self.a = a
        self.b = b
        self.diam = diam
        self.max_z = 10.
        self.min_z = 0.
        self.L_r_L = L_r_L.reshape([3, 1])
        self.DCM_SL = rbk.euler2122C(e212)
        self.set_limits()
        self.S_focus = np.zeros(3)
        self.L_focus = np.zeros(3)
        self.set_focus()
        self.name = 'secondary'

    def set_focus(self):
        f = np.sqrt(self.a**2 + self.b**2)
        self.S_focus = np.array([0., 0., f])
        self.L_focus = np.dot(self.DCM_SL.transpose(), self.S_focus) + self.L_r_L.reshape([3, ])
        return

    def set_limits(self):
        rad = self.diam / 2.
        self.max_z = self.equation(rad, 0)
        self.min_z = self.a
        return

    def surface_points(self, n=50):
        xs = np.linspace(-self.diam / 2., self.diam / 2., n).tolist()
        x_out = []
        ys = np.linspace(-self.diam / 2., self.diam / 2., n).tolist()
        y_out = []
        z_out = []
        for x in xs:
            for y in ys:
                x_out.append(x)
                y_out.append(y)
                z_out.append(self.equation(x, y))
        x_out = np.array(x_out)
        y_out = np.array(y_out)
        z_out = np.array(z_out)
        x_out = x_out[(z_out <= self.max_z) & (z_out >= self.min_z)]
        y_out = y_out[(z_out <= self.max_z) & (z_out >= self.min_z)]
        z_out = z_out[(z_out <= self.max_z) & (z_out >= self.min_z)]
        X = np.vstack([x_out, y_out, z_out])
        X = np.dot(self.DCM_SL.transpose(), X)
        X = X + self.L_r_L
        return X

    def surface_mesh(self):
        out_rad = self.diam / 2.
        rad_points = np.linspace(0., out_rad, 2)
        theta_points = np.linspace(0., np.pi * 2., 30)
        R, T = np.meshgrid(rad_points, theta_points)
        X, Y = R * np.cos(T), R * np.sin(T)
        Z = self.equation(X, Y)
        for i in range(np.shape(Z)[0]):
            for j in range(np.shape(Z)[1]):
                vec = np.array([X[i,j], Y[i,j], Z[i,j]])
                vec = np.dot(self.DCM_SL.transpose(), vec)
                vec = vec + self.L_r_L.reshape(3,)
                X[i,j], Y[i,j], Z[i, j] = vec[0], vec[1], vec[2]
        return X, Y, Z

    def equation(self, xs, ys):
        # defining equation for a circular hyperboloid
        return np.sqrt(((xs**2 + ys**2)/self.b ** 2 + 1) * self.a**2)

    def intersect_rays(self, L_X_0, L_X_d):
        # takes in ray starts and direction unit vectors in lab frame
        num = np.shape(L_X_0)[1]
        S_X_0 = np.dot(self.DCM_SL, L_X_0 - self.L_r_L)
        S_X_d = np.dot(self.DCM_SL, L_X_d)
        x0s = S_X_0[0, :]
        xds = S_X_d[0, :]
        y0s = S_X_0[1, :]
        yds = S_X_d[1, :]
        z0s = S_X_0[2, :]
        zds = S_X_d[2, :]
        A = (xds**2 + yds**2) / self.b**2 - zds**2 / self.a**2
        B = 2 * (x0s*xds + yds*y0s) / self.b**2 - 2 * z0s * zds / self.a**2
        C = (x0s**2 + y0s**2) / self.b**2 - z0s**2 / self.a**2 + 1.
        non_nan = ~np.isnan(x0s)
        ts = mullers_quadratic_equation(A[non_nan], B[non_nan], C[non_nan], 1)
        S_X_1 = S_X_0
        S_X_1[:, non_nan] = S_X_0[:, non_nan] + ts * S_X_d[:, non_nan]
        L_X_1 = np.dot(self.DCM_SL.transpose(), S_X_1) + self.L_r_L
        return L_X_1

    def normal(self, L_X):
        # given points on the hyperbolloid, gives unit normal vectors.
        S_X = np.dot(self.DCM_SL, L_X - self.L_r_L)
        xs = S_X[0, :]
        ys = S_X[1, :]
        zs = S_X[2, :]
        num = np.shape(ys)
        S_N = np.array([2 * xs / self.b ** 2, 2 * ys / self.b ** 2, -2 * zs / self.a**2])
        S_N_hat = S_N / np.linalg.norm(S_N, axis=0)
        L_N_hat = np.dot(self.DCM_SL.transpose(), S_N_hat)
        return L_N_hat

    def reflect_rays(self, L_X, L_d_i):
        # takes an intersect point L_X and incoming direction L_d_i
        # produces an outgoing direction L_d_o
        num = np.shape(L_X)[1]
        L_N_hat = self.normal(L_X)
        L_d_o = []
        for i in range(num):
            incoming = L_d_i[:,i]
            nHat = L_N_hat[:,i]
            M = np.eye(3) - 2 * np.outer(nHat, nHat)
            L_d_o.append(np.dot(M, incoming))
        return np.array(L_d_o).transpose()  # just put it back into 3xN array format

    def miss_rays(self, L_X):
        # L_X is an intersection point for a hyperboloid that is infinite with no holes
        S_X = np.dot(self.DCM_SL, L_X - self.L_r_L)
        zs = S_X[2, :]
        zs[np.isnan(zs)] = 1E10
        temp = S_X.transpose()
        temp[zs > self.max_z] = np.array([np.nan] * 3)
        S_X = temp.transpose()
        L_X = np.dot(self.DCM_SL.transpose(), S_X) + self.L_r_L
        return L_X

class FlatImagePlane:
    def __init__(self, w, h, L_r_L, e212):
        self.w = w
        self.h = h
        self.L_r_L = L_r_L.reshape([3, 1])
        self.DCM_SL = rbk.euler2122C(e212)
        self.name = "image_plane"

    def intersect_rays(self, L_X_0, L_X_d):
        # takes in ray starts and direction unit vectors in lab frame
        num = np.shape(L_X_0)[1]
        S_X_0 = np.dot(self.DCM_SL, L_X_0 - self.L_r_L)
        S_X_d = np.dot(self.DCM_SL, L_X_d)
        x0s = S_X_0[0, :]
        xds = S_X_d[0, :]
        y0s = S_X_0[1, :]
        yds = S_X_d[1, :]
        z0s = S_X_0[2, :]
        zds = S_X_d[2, :]
        ts = -z0s / zds
        xs = x0s + ts * xds
        ys = y0s + ts * yds
        S_X_1 = S_X_0 + ts * S_X_d
        temp = S_X_1.transpose()
        xs[np.isnan(xs)] = 10000.
        ys[np.isnan(ys)] = 10000.
        temp[(np.fabs(xs) > self.w / 2.) | (np.fabs(ys) > self.h / 2.)] = np.array([np.nan] * 3)
        S_X_1 = temp.transpose()
        L_X_1 = np.dot(self.DCM_SL.transpose(), S_X_1) + self.L_r_L
        return L_X_1

    def miss_rays(self, L_X):
        return L_X

    def reflect_rays(self, L_X, L_d):
        return L_d

    def extract_image(self, L_X):
        # takes in the points that intersected the plane
        image = np.dot(self.DCM_SL, L_X - self.L_r_L)[0:2, :]
        return image

    def surface_points(self, num=15):
        xs = np.linspace(-self.w/2, self.w/2)
        ys = np.linspace(-self.h/2, self.h/2)
        L_pt_list = []
        for x in xs:
            for y in ys:
                S_pt = np.array([x, y, 0])
                L_pt = (np.dot(self.DCM_SL.transpose(), S_pt) + self.L_r_L.reshape([3, ])).transpose()
                L_pt_list.append(L_pt)
        return np.array(L_pt_list).transpose()

    def surface_mesh(self):
        X = np.linspace(-self.w / 2., self.w / 2.)
        Y = np.linspace(-self.h / 2., self.h / 2.)
        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(np.shape(X))
        for i in range(np.shape(Z)[0]):
            for j in range(np.shape(Z)[1]):
                vec = np.array([X[i,j], Y[i,j], Z[i,j]])
                vec = np.dot(self.DCM_SL.transpose(), vec)
                vec = vec + self.L_r_L.reshape(3,)
                X[i,j], Y[i,j], Z[i, j] = vec[0], vec[1], vec[2]
        return X, Y, Z

    def plot_image(self, ax, L_x):
        pts = self.extract_image(L_x)
        ax.scatter(pts[0, :], pts[1, :], color='black', s=1)
        max_x = np.max(pts[0, :][~np.isnan(pts[0, :])])
        min_x = np.min(pts[0, :][~np.isnan(pts[0, :])])
        max_y = np.max(pts[1, :][~np.isnan(pts[1, :])])
        min_y = np.min(pts[1, :][~np.isnan(pts[1, :])])
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        return

class SphericalDetector:
    # a spherical detector with square edges. it's a square with spherical curvature.
    def __init__(self):
        self.r = 1.
        self.w = 1.  # width/height of detector
        self.L_r_L = np.zeros(3)
        self.DCM_SL = np.eye(3)
        self.name = "image_plane"

    def equation(self, xs, ys):
        # defining equation for a sphere
        return np.sqrt(self.r**2 - xs**2 - ys**2)

    def intersect_rays(self, L_X_0, L_X_d):
        # takes in ray starts and direction unit vectors in lab frame
        S_X_0 = np.dot(self.DCM_SL, L_X_0 - self.L_r_L)
        S_X_d = np.dot(self.DCM_SL, L_X_d)
        x0s = S_X_0[0, :]
        xds = S_X_d[0, :]
        y0s = S_X_0[1, :]
        yds = S_X_d[1, :]
        z0s = S_X_0[2, :]
        zds = S_X_d[2, :]
        A = xds**2 + yds**2 + zds**2
        B = 2 * (x0s*xds + yds*y0s + zds*z0s)
        C = x0s**2 + y0s**2 + z0s**2 - self.r**2
        non_nan = ~np.isnan(x0s)
        ts = mullers_quadratic_equation(A[non_nan], B[non_nan], C[non_nan], -1)
        S_X_1 = S_X_0
        S_X_1[:, non_nan] = S_X_0[:, non_nan] + ts * S_X_d[:, non_nan]
        L_X_1 = np.dot(self.DCM_SL.transpose(), S_X_1) + self.L_r_L
        return L_X_1

    def miss_rays(self, L_X):
        return L_X

    def reflect_rays(self, L_X, L_d):
        return L_d

    def extract_image(self, L_X):
        # takes in the points that intersected the detector
        # spits out angular coordinate on detector
        S_X = np.dot(self.DCM_SL, L_X - self.L_r_L)
        xs = S_X[0, :]
        ys = S_X[1, :]
        zs = S_X[2, :]
        ras = np.arctan(ys / (zs + self.r))
        decs = np.arctan(xs / (zs + self.r))
        ra_dec = np.vstack([ras, decs])
        return ra_dec

    def surface_mesh(self):
        X = np.linspace(-self.w / 2., self.w / 2.)
        Y = np.linspace(-self.w / 2., self.w / 2.)
        X, Y = np.meshgrid(X, Y)
        Z = self.equation(X, Y)
        for i in range(np.shape(Z)[0]):
            for j in range(np.shape(Z)[1]):
                vec = np.array([X[i,j], Y[i,j], Z[i,j]])
                vec = np.dot(self.DCM_SL.transpose(), vec)
                vec = vec + self.L_r_L.reshape(3,)
                X[i,j], Y[i,j], Z[i, j] = vec[0], vec[1], vec[2]
        return X, Y, Z

def prep_rays_for_plot(ray_history):
    ray_list = []
    num_rays = np.shape(ray_history[-1])[1]
    ray_array = np.array(ray_history)
    for i in range(num_rays):
        xs = ray_array[:, 0, i]
        ys = ray_array[:, 1, i]
        zs = ray_array[:, 2, i]
        ray_list.append([xs, ys, zs])
    return ray_list

class AngledCircleRayDef:
    def __init__(self):
        self.rad = 0.5
        self.angles = [0.]
        self.num_circ = 15
        self.per_circ = 150

def make_angled_circle_rays(inputs):
    rad_inc = inputs.rad / inputs.num_circ
    theta_inc = np.pi * 2 / inputs.per_circ
    rays_list = []
    rays_d_list = []
    for angle in inputs.angles:
        rays = []
        angle = angle / 3600. * np.pi / 180.
        for i in range(inputs.num_circ):
            for j in range(inputs.per_circ):
                r = rad_inc * i
                x, y, z = 0., r * np.cos(theta_inc * j), r * np.sin(theta_inc * j)
                rays.append(np.array([x, y, z]))
        rays = np.array(rays).transpose()
        ray_dirs = np.array([np.array([1, 0, 0])] * np.shape(rays)[1]).transpose()
        DCM = rbk.euler1232C([0., 0., angle]).transpose()
        ray_dirs = np.dot(DCM, ray_dirs)
        rays_list.append(rays)
        rays_d_list.append(ray_dirs)
    return rays_list, rays_d_list

def make_one_edge_ray(rad, angle):
    x, y, z = 0., rad, 0.,
    L_X = np.array([x,y,z]).reshape([3, 1])
    angle = angle/3600. * np.pi/180.
    dir = np.array([np.cos(angle), -np.sin(angle), 0]).reshape([3, 1])
    return L_X, dir

def cassegrain_set_up(inputs):
    f_1 = inputs.f_num_1 * inputs.d_1
    M = inputs.f_num_total / inputs.f_num_1
    alpha = inputs.e / f_1
    beta = (M - alpha) / (M + 1.)
    c = 1. - beta
    d = -beta * f_1  # separation of primary and secondary
    d_2 = c * inputs.d_1
    f_2 = (f_1 + inputs.e) / 2.
    secondary_a = f_2 - (f_1 + d)
    secondary_b = np.sqrt(f_2 ** 2 - secondary_a ** 2)
    primary_position = np.array([inputs.primary_x, 0, 0])
    primary_a = 1. / (4. * f_1)
    primary_hole_diam = 0.1
    primary = parabolicMirrorWithHole(primary_a, inputs.d_1, primary_hole_diam, inputs.orientation_212, primary_position)
    secondary_position = np.array([inputs.primary_x + d + secondary_a, 0., 0])
    secondary = ConvexHyperbolicMirror(secondary_b, secondary_a, d_2, secondary_position, inputs.orientation_212)
    dead_spot_position = secondary_position - np.array([secondary.max_z, 0., 0.])
    dead_spot = circleOfDeath(d_2 / 2., dead_spot_position, inputs.orientation_212)
    image_plane_position = primary_position + np.array([inputs.e + inputs.focal_plane_offset, 0., 0.])
    image_plane = FlatImagePlane(inputs.d_1, inputs.d_1, image_plane_position, inputs.orientation_212)
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
    primary = parabolicMirrorWithHole(primary_a, d_1, primary_hole_diam, orientation, primary_position)
    secondary_position = np.array([primary_x + d + secondary_a, 0., 0])
    secondary = ConvexHyperbolicMirror(secondary_b, secondary_a, d_2, secondary_position, orientation)
    dead_spot_position = secondary_position - np.array([secondary.max_z, 0., 0.])
    dead_spot = circleOfDeath(d_2 / 2., dead_spot_position, orientation)
    image_plane_position = primary_position + np.array([e+image_plane_offset, 0., 0.])
    image_plane = SphericalImagePlane(image_plane_r, image_plane_max_r, image_plane_position, orientation)
    return [dead_spot, primary, secondary, image_plane]

def scatter_surfaces(ax, surfaces):
    for i, surf in zip(range(len(surfaces)), surfaces):
        pts = surf.surface_points()
        ax.scatter(pts[0, :], pts[1, :], pts[2, :], label=surf.name, s=1)
    return

def plot_surfaces(ax, surfaces):
    for i, surf in zip(range(len(surfaces)), surfaces):
        X, Y, Z = surf.surface_mesh()
        ax.plot_surface(X, Y, Z, alpha=0.4)
    return

def plot_rays(ax, ray_hist, al=1.0):
    ray_hist = np.array(ray_hist)
    for r in range(np.shape(ray_hist)[2]):
            ray = ray_hist[:, :, r]
            ax.plot(ray[:, 0], ray[:, 1], ray[:, 2], color='black', alpha=al)
    return

def save_3d_plot(surfaces, ray_hist, path='/Users/sqc0815/Not_Backed_Up/hwRepo/ASTR5760', alpha=.015):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    plot_surfaces(ax, surfaces)
    plot_rays(ax, ray_hist, alpha)
    plt.title('3-D View of Ray Trace')
    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    ax.set_zlabel('z [m]')
    plt.legend()
    # ax.set_aspect('equal')
    #plt.savefig(path + "/figures/3Dview.png")
    return fig

def plot_ray_starts(xs, ys, angle, path='/Users/sqc0815/Not_Backed_Up/hwRepo/ASTR5760'):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.scatter(-ys, -xs, s=2)
    ax.set_ylim([-0.5, 0.5])
    ax.set_xlim([-0.5, 0.5])
    ax.set_title('ray starts (inverted)')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    plt.savefig(path + '/figures/offAxisRayStart' + str(angle) + '.png')
    return fig

def plot_flat_image(image_plane, L_X, angle, path='/Users/sqc0815/Not_Backed_Up/hwRepo/ASTR5760'):
    img = plt.figure()
    plt.clf()
    ax = img.add_subplot(111)
    pts = image_plane.extract_image(L_X)
    ax.scatter(pts[0, :], pts[1, :], label='image', color='black', s=1)
    max_x = np.max(pts[0, :][~np.isnan(pts[0, :])])
    min_x = np.min(pts[0, :][~np.isnan(pts[0, :])])
    max_y = np.max(pts[1, :][~np.isnan(pts[1, :])])
    min_y = np.min(pts[1, :][~np.isnan(pts[1, :])])
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('flat plane image at ' + str(angle) + "\'\'")
    # ax.set_aspect('equal')
    plt.savefig(path + '/figures/image_' + str(angle) + '.png')
    return img

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

def plot_flat_image_at_dist(image_plane, L_X, angle, offset, path='/Users/sqc0815/Not_Backed_Up/hwRepo/ASTR5760'):
    img = plt.figure()
    plt.clf()
    ax = img.add_subplot(111)
    pts = image_plane.extract_image(L_X) * 1000
    ax.scatter(pts[0, :], pts[1, :], label='image', color='black', s=1)
    max_x = np.max(pts[0, :][~np.isnan(pts[0, :])])
    min_x = np.min(pts[0, :][~np.isnan(pts[0, :])])
    max_y = np.max(pts[1, :][~np.isnan(pts[1, :])])
    min_y = np.min(pts[1, :][~np.isnan(pts[1, :])])
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_title('flat plane image at ' + str(angle) + "\'\'_" + str(offset))
    plt.savefig(path + '/figures/image_' + str(angle) + '_' + str(offset) + '.png')
    return img

class CassegrainDefinition:
    def __init__(self):
        self.f_num_1 = 3.
        self.d_1 = 1.
        self.f_num_total = 15.
        self.e = 0.15
        self.primary_x = 3.
        self.orientation_212 = [-np.pi / 2., 0, 0]
        self.focal_plane_offset = 0.
        return

class Instrument:
    def __init__(self):
        self.surfaces = []
        self.detector = None
        return

    def set_surfaces(self, surfs):
        self.surfaces = surfs
        self.detector = surfs[-1]
        return

    def set_detector(self, detector):
        self.detector = detector
        self.surfaces[-1] = self.detector

class RayTraceResults:
    def __init__(self):
        self.image = None
        self.rms = 0.0
        self.mean_spread = 0.0
        return

    def find_image_rms(self):
        xs, ys = self.image[0], self.image[1]
        N = len(xs)
        mean_x, mean_y = np.nanmean(xs), np.nanmean(ys)
        self.rms = np.sqrt(np.nansum((xs - mean_x)**2 + (ys - mean_y)**2) / N)
        return

    def find_image_spread(self):
        xs, ys = self.image[0], self.image[1]
        spread_x, spread_y = (np.nanmax(xs) - np.nanmin(xs)), (np.nanmax(ys) - np.nanmin(ys))
        self.mean_spread = np.nanmean([spread_x, spread_y])
        return

class Experiment:
    def __init__(self):
        self.instrument = None # the instrument to send rays through including detector
        self.L_ray_pts = None  # current location of rays in lab frame, L in [m]
        self.L_ray_dir = None  # direction of rays in lab frame, L
        self.ray_hist = []
        self.name = "experiment"
        self.L_ray_starts = None
        self.L_ray_start_dirs = None
        return

    def add_instrument(self, inst_in):
        self.instrument = inst_in
        return

    def set_ray_starts(self, ray_starts):
        self.L_ray_pts = ray_starts
        self.ray_hist.append(ray_starts)  # will I have to np.copy here?
        self.L_ray_starts = ray_starts  # can reset to this
        return

    def set_ray_start_dir(self, ray_dirs):
        self.L_ray_dir = ray_dirs
        self.L_ray_start_dirs = ray_dirs  # can reset to this
        return

    def reset(self):
        self.L_ray_pts = self.L_ray_starts
        self.ray_hist = [self.L_ray_pts]
        self.L_ray_dir = self.L_ray_start_dirs
        return

    def trace_rays(self):
        for surf in self.instrument.surfaces:
            self.L_ray_pts = surf.intersect_rays(self.L_ray_pts, self.L_ray_dir)
            self.L_ray_pts = surf.miss_rays(self.L_ray_pts)
            self.L_ray_dir = surf.reflect_rays(self.L_ray_pts, self.L_ray_dir)
            self.ray_hist.append(self.L_ray_pts)  # will I have to np.copy here?
        return

    def run(self):
        self.trace_rays()
        result = RayTraceResults()
        result.image = self.instrument.detector.extract_image(self.L_ray_pts)
        result.find_image_rms()
        result.find_image_spread()
        return result

    def result_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot('111')
        ax.set_title('Results from ' + self.name)
        self.instrument.detector.plot_image(ax, self.L_ray_pts)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        return fig

    def spherical_result_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot('111')
        ax.set_title('Results from ' + self.name)
        self.instrument.detector.plot_image(ax, self.L_ray_pts)
        ax.set_xlabel('RA [rad]')
        ax.set_ylabel('DEC [rad]')
        return fig


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

class RowlandCircle:
    # a rowland circle diffraction grating
    # circular curvature, but square cut edges
    # the surface normal always points in +z direction in local (S) frame at the origin
    # i.e., the surface is grazing tangent to the origin (the sphere is not centered on the S origin
    # the S frame y-axis is parallel to the gratings at the origin
    # the x-axis is y cross z, then
    # The sphere is centered at 0, 0, 0
    def __init__(self):
        self.r = 1.  # radius of Rowland Circle
        self.w = 0.25  # it's a square chunk of a circle, this is the side length.
        self.m = 1  # diffraction order. For now, just choose one order to look into and do it multiple times to get
                    # multiple orders
        self.lam = 1000. / 1E10  # wavelength to consider. Again, this is just a hack for now
        self.line_density = 1000.  # grating line density
        self.DCM_SL = np.eye(3)  # rotation into the surface frame from lab frame
        self.L_r_L = np.zeros(3).reshape([3, 1])  # offset from lab origin in lab frame coordinates
        self.grating_direction_q = np.array([0., -1., 0.])  # parallel to gratings
        self.unprojected_spacing = 1.
        self.central_normal = np.array([0., 0., 1.]).reshape([3, 1])
        return

    def set_radius(self, radius):
        self.r = radius
        return

    def set_line_density(self, density):
        # lines per mm
        self.line_density = density  # per mm
        self.unprojected_spacing = 1. / (1000. * self.line_density)  # [m / space]
        return

    def set_position(self, pos):
        self.L_r_L = pos
        return

    def set_DCM(self, dcm):
        self.DCM_SL = dcm
        return

    def set_width(self, width):
        self.w = width
        return

    def set_order(self, order):
        self.m = order
        return

    def set_wavelength(self, wavelength):
        self.lam = wavelength / 1E10  # convert angstrom to m
        return

    def surface_mesh(self):
        X = np.linspace(-self.w / 2., self.w / 2.)
        Y = np.linspace(-self.w / 2., self.w / 2.)
        X, Y = np.meshgrid(X, Y)
        Z = self.equation(X, Y)
        for i in range(np.shape(Z)[0]):
            for j in range(np.shape(Z)[1]):
                vec = np.array([X[i,j], Y[i,j], Z[i,j]])
                vec = np.dot(self.DCM_SL.transpose(), vec)
                vec = vec + self.L_r_L.reshape(3,)
                X[i, j], Y[i, j], Z[i, j] = vec[0], vec[1], vec[2]
        return X, Y, Z

    def equation(self, xs, ys):
        # defining equation for a sphere
        num = np.shape(xs)[0]
        return solve_quadratic(np.ones(num), -2. * self.r * np.ones(num), xs**2 + ys**2, -1)

    def intersect_rays(self, L_X_0, L_X_d):
        # takes in ray starts and direction unit vectors in lab frame
        S_X_0 = np.dot(self.DCM_SL, L_X_0 - self.L_r_L)
        S_X_d = np.dot(self.DCM_SL, L_X_d)
        x0s = S_X_0[0, :]
        xds = S_X_d[0, :]
        y0s = S_X_0[1, :]
        yds = S_X_d[1, :]
        z0s = S_X_0[2, :]
        zds = S_X_d[2, :]
        A = xds**2 + yds**2 + zds**2
        B = 2. * (x0s*xds + yds*y0s + zds*z0s - self.r*zds)
        C = x0s**2 + y0s**2 + z0s**2 - 2 * self.r * z0s
        non_nan = ~np.isnan(x0s)
        ts = solve_quadratic(A[non_nan], B[non_nan], C[non_nan], 1)
        S_X_1 = S_X_0
        S_X_1[:, non_nan] = S_X_0[:, non_nan] + ts * S_X_d[:, non_nan]
        L_X_1 = np.dot(self.DCM_SL.transpose(), S_X_1) + self.L_r_L
        return L_X_1

    def normal(self, L_X):
        # given points on the sphere, gives unit normal vectors.
        S_X = np.dot(self.DCM_SL, L_X - self.L_r_L)
        xs = S_X[0, :]
        ys = S_X[1, :]
        zs = S_X[2, :]
        S_N = -np.array([2. * xs, 2. * ys, 2. * (zs - self.r)])
        S_N_hat = S_N / np.linalg.norm(S_N, axis=0)
        L_N_hat = np.dot(self.DCM_SL.transpose(), S_N_hat)
        return L_N_hat

    def miss_rays(self, L_X):
        # based on width of the element. misses if outside x,y bounds based on width of grating
        S_X = np.dot(self.DCM_SL, L_X - self.L_r_L)
        xs = np.fabs(S_X[0, :])
        xs[np.isnan(xs)] = -1.
        ys = np.fabs(S_X[1, :])
        ys[np.isnan(ys)] = -1.
        temp = S_X.transpose()
        temp[(xs > self.w / 2.) | (ys > self.w / 2.)] = np.array([np.nan] * 3)
        S_X = temp.transpose()
        L_X = np.dot(self.DCM_SL.transpose(), S_X) + self.L_r_L
        return L_X

    def reflect_rays(self, L_X, L_d):
        # following spencer and murty, 1961
        S_d = np.dot(self.DCM_SL, L_d)
        r = np.dot(self.DCM_SL, self.normal(L_X))  # normal vectors at every intersection point in S frame
        Ks = r[0, :]
        Ls = r[1, :]
        Ms = r[2, :]
        u = 1. / np.sqrt(1. + Ks**2. / (Ls**2. + Ms**2.))
        v = -Ks * Ls * u / (Ls**2 + Ms**2)
        w = -Ks * Ms * u / (Ls**2 + Ms**2)
        p = np.vstack([u, v, w])
        d = self.unprojected_spacing / u
        L = self.m * self.lam / d  # assumes wavelength is defined in current medium or probably that we're always in vacuum
        mu = 1.  # no change in medium
        kulvmw = np.array([np.dot(p[:, i], S_d[:, i]) for i in range(np.shape(p)[1])])
        b_prime = (mu**2. - 1. + L**2. - 2. * mu * L * kulvmw)  # notice I don't divide by r**2 because I norm my normals
        a = mu * np.array([np.dot(S_d[:, i], r[:, i]) for i in range(np.shape(S_d)[1])])
        nans = np.isnan(b_prime)
        doable = np.ones(len(b_prime), dtype=bool)
        doable[~nans] = b_prime[~nans] <= a[~nans]**2
        doable[nans] = False
        G = (solve_quadratic(np.ones(np.sum(doable)), 2. * a[doable], b_prime[doable], 1))
        S_d[:, doable] = S_d[:, doable] - L[doable] * p[:, doable] + G.flatten() * r[:, doable]
        S_d[:, ~doable] = np.ones([3, np.sum(~doable)]) * np.nan
        L_d = np.dot(self.DCM_SL.transpose(), S_d)
        return L_d

class CylindricalDetector:
    # a cylindrical (circular extrusion) detector.
    # The long axis is the x-axis
    # an axial ray would come in the z^ axis, going in the negative x^ direction. a reflected ray would go off in the z^
    # y is radial. orthogonal to x. orthogonal to z (which is also radial).
    # The cylinder is lifted on the z-axis so that the "vertex" of the surface is at the frame origin
    def __init__(self):
        self.r = 1.
        self.h = 1.  # width/height of detector
        self.sweep = np.pi / 8.
        self.y_min = -100.
        self.y_max = 100.
        self.set_y_limits()
        self.L_r_L = np.zeros(3)
        self.DCM_SL = np.eye(3)
        self.name = "image_plane"

    def set_y_limits(self):
        self.y_max = self.r * np.sin(self.sweep / 2.)
        self.y_min = -self.y_max
        return

    def equation(self, xs, ys):
        # defining equation for a cylinder
        # return the z-value of the surface
        num = np.shape(xs)[0]
        return solve_quadratic(np.ones(num), -2 * self.r, ys ** 2, -1)

    def set_radius(self, r):
        self.r = r
        return

    def set_height(self, h):
        self.h = h
        return

    def set_sweep_angle(self, ang):
        self.sweep = ang
        return

    def set_position(self, pos):
        self.L_r_L = pos
        return

    def set_DCM(self, dcm):
        self.DCM_SL = dcm
        return

    def normal(self, L_X):
        # given points on the cylinder, gives unit normal vectors.
        S_X = np.dot(self.DCM_SL, L_X - self.L_r_L)
        ys = S_X[1, :]
        zs = S_X[2, :]
        num = np.shape(zs)[0]
        S_N = -np.array([np.zeros(num), 2 * ys, 2 * (zs - self.r)])
        S_N_hat = S_N / np.linalg.norm(S_N, axis=0)
        L_N_hat = np.dot(self.DCM_SL.transpose(), S_N_hat)
        return L_N_hat

    def intersect_rays(self, L_X_0, L_X_d):
        # takes in ray starts and direction unit vectors in lab frame
        S_X_0 = np.dot(self.DCM_SL, L_X_0 - self.L_r_L)
        S_X_d = np.dot(self.DCM_SL, L_X_d)
        y0s = S_X_0[1, :]
        yds = S_X_d[1, :]
        z0s = S_X_0[2, :]
        zds = S_X_d[2, :]
        A = zds**2 + yds**2
        B = 2 * z0s * zds - 2 * self.r * zds + 2 * y0s * yds
        C = z0s**2 - 2 * self.r * z0s + y0s**2
        non_nan = ~np.isnan(y0s)
        ts = solve_quadratic(A[non_nan], B[non_nan], C[non_nan], 1)
        S_X_1 = S_X_0
        S_X_1[:, non_nan] = S_X_0[:, non_nan] + ts * S_X_d[:, non_nan]
        L_X_1 = np.dot(self.DCM_SL.transpose(), S_X_1) + self.L_r_L
        return L_X_1

    def miss_rays(self, L_X):
        S_X = np.dot(self.DCM_SL, L_X - self.L_r_L)
        xs = S_X[0, :]
        ys = S_X[1, :]
        xs[np.isnan(xs)] = -1E6
        ys[np.isnan(ys)] = -1E6
        temp = S_X.transpose()
        temp[(ys > self.y_max) | (ys < self.y_min) | (xs < -self.h / 2.) | (xs > self.h / 2.)] = np.array([np.nan] * 3)
        S_X = temp.transpose()
        L_X = np.dot(self.DCM_SL.transpose(), S_X) + self.L_r_L
        return L_X

    def reflect_rays(self, L_X, L_d):
        return L_d

    def extract_image(self, L_X):
        # should change this into a linear and angular  coordinate. Right now it does a spherical RA/DEC transformation
        S_X = np.dot(self.DCM_SL, L_X - self.L_r_L)
        xs = S_X[0, :]
        ys = S_X[1, :]
        RA = - np.arcsin(ys / self.r)
        ra_h = np.vstack([RA, xs])
        return ra_h

    def surface_mesh(self):
        X = np.linspace(-self.h / 2., self.h / 2.)
        angles = np.linspace(-self.sweep / 2., self.sweep / 2.)
        Y = np.sin(angles) * self.r
        X, Y = np.meshgrid(X, Y)
        Z = self.equation(X, Y)
        for i in range(np.shape(Z)[0]):
            for j in range(np.shape(Z)[1]):
                vec = np.array([X[i,j], Y[i,j], Z[i,j]])
                vec = np.dot(self.DCM_SL.transpose(), vec)
                vec = vec + self.L_r_L.reshape(3,)
                X[i,j], Y[i,j], Z[i, j] = vec[0], vec[1], vec[2]
        return X, Y, Z




