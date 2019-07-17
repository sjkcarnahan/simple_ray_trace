'''
Scott Carnahan
simple ray trace - Experiment and Results Containers
Spring 2019
'''

import numpy as np
import matplotlib.pyplot as plt
import light_sources as ls

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

    def trace_rays_test(self):
        for i, surf in enumerate(self.instrument.surfaces):
            if i in [0, 1, 2]:
                rays = ls.Ray(self.L_ray_pts, self.L_ray_dir)
                rays = surf.interact(rays)
                self.L_ray_pts = rays.X
                self.L_ray_dir = rays.d
            else:
                self.L_ray_pts = surf.intersect_rays(self.L_ray_pts, self.L_ray_dir)
                self.L_ray_pts = surf.miss_rays(self.L_ray_pts)
                self.L_ray_dir = surf.reflect_rays(self.L_ray_pts, self.L_ray_dir)
            self.ray_hist.append(self.L_ray_pts)
        return

    def run(self):
        self.trace_rays_test()
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



