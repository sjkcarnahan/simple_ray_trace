'''
Scott Carnahan
ASTR 5760 Proj 1 - Ray Trace
Spring 2019
'''

import numpy as np
import matplotlib.pyplot as plt
from modules import rayTracing as rt
import project_inputs as pi

# plotting flags
show_plots = True
save_plots = True

# create an Experiment
exp_1 = rt.Experiment()
exp_1.name = "basic_trace"

# bring in the instrument
exp_1.add_instrument(pi.cass)

# Make rays for this Experiment
exp_1.set_ray_starts(pi.basical_paraxial_rays.X)
exp_1.set_ray_start_dir(pi.basical_paraxial_rays.d)

# run and plot
exp_1.run()
result_plot = exp_1.result_plot()

if save_plots:
    plt.figure(result_plot.number)
    plt.savefig('./figures/basic_trace/basicTrace.png')
if show_plots:
    plt.show()
plt.close('all')

# final notes: maybe just add the 3D Experiment set up plot back in? Or do that separately

