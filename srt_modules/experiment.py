'''
Scott Carnahan
simple ray trace - Experiment and Results Containers
Spring 2019
'''

import numpy as np
import matplotlib.pyplot as plt
import light_sources as ls
from copy import deepcopy

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
        self.rays = ls.Ray()
        self.ray_hist = [np.array([])]  # list of ray positions on each surface
        self.name = "experiment"
        self.ray_starts = ls.Ray()
        return

    def add_instrument(self, inst_in):
        self.instrument = inst_in
        return

    def set_ray_starts(self, ray_starts):
        self.rays = deepcopy(ray_starts)
        self.ray_hist.append(ray_starts.X)  # will I have to np.copy here?
        self.ray_starts = deepcopy(ray_starts)
        return

    def reset(self):
        self.rays = deepcopy(self.ray_starts)
        self.ray_hist = [self.rays.X]
        return

    def trace_rays(self):
        for i, surf in enumerate(self.instrument.surfaces):
            self.rays = surf.interact(self.rays)
            self.ray_hist.append(self.rays.X)
        return

    def run(self):
        self.trace_rays()
        result = RayTraceResults()
        result.image = self.instrument.detector.extract_image()
        result.find_image_rms()
        result.find_image_spread()
        return result

    def result_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot('111')
        ax.set_title('Results from ' + self.name)
        self.instrument.detector.plot_image(ax)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        return fig

    def spherical_result_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot('111')
        ax.set_title('Results from ' + self.name)
        self.instrument.detector.plot_image(ax, self.rays.X)
        ax.set_xlabel('RA [rad]')
        ax.set_ylabel('DEC [rad]')
        return fig



