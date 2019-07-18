'''
Scott Carnahan
Spring 2019
This example traces paraxial rays from some start to the image plane through a basic cassegrain telescope.
The resulting plot should have exceedingly small dimensions that represent the machine error in tracing the rays
 to a focus on the image plane.
'''

import matplotlib.pyplot as plt
from srt_modules import experiment as rt
from srt_instances import instrument_instances as ii, ray_instances as ri

# plotting flags
show_plots = True
save_plots = True

# create an Experiment
exp_1 = rt.Experiment()  # a container to hold the instrument and results etc. and run tests
exp_1.name = "basic_trace"  # this can be used in file names when saving things

# bring in the instrument
exp_1.add_instrument(ii.cass)  # the instrument is defined in a different file to keep it simple and clean here

# Make rays for this Experiment. In rendering lingo this is defining a light source, or multiple point light sources
exp_1.set_ray_starts(ri.basic_paraxial_rays)

# run and plot
results = exp_1.run()  # steps through all surfaces to find ray intersections and reflections
result_plot = exp_1.result_plot()

if save_plots:
    plt.figure(result_plot.number)
    plt.savefig('./figures/basic_trace/basicTrace.png')
if show_plots:
    plt.show()
plt.close('all')