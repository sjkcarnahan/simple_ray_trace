'''
Scott Carnahan
ASTR 5760 Proj 1 - Ray Trace
Spring 2019
'''

import matplotlib.pyplot as plt
from modules import rayTracing as rt
import project_inputs as pi

# plotting flags
show_plots = True
save_plots = True

# create an Experiment
exp_1 = rt.Experiment()  # a container to hold the instrument and results etc. and run tests
exp_1.name = "basic_trace"  # this can be used in file names when saving things

# bring in the instrument
exp_1.add_instrument(pi.cass)  # the instrument is defined in a different file to keep it simple and clean here

# Make rays for this Experiment. In rendering lingo this is defining a light source, or multiple point light sources
exp_1.set_ray_starts(pi.basical_paraxial_rays.X)
exp_1.set_ray_start_dir(pi.basical_paraxial_rays.d)

# run and plot
exp_1.run()  # steps through all surfaces to find ray intersections and reflections
result_plot = exp_1.result_plot()

if save_plots:
    plt.figure(result_plot.number)
    plt.savefig('./figures/basic_trace/basicTrace.png')
if show_plots:
    plt.show()
plt.close('all')

