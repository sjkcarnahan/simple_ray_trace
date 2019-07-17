'''
Scott Carnahan
This file shows how to adjust the image plane position to account for non-paraxial rays.
The result should be a plot with a minimum trough indicating the best offset for the image plane relative to
the paraxial focus in [um] along the instrument axis.
'''

import numpy as np
import matplotlib.pyplot as plt
from srt_modules import experiment
from srt_instances import instrument_instances as ii, ray_instances as ri

show_plots = True
save_plots = True
make_plots = show_plots or save_plots

# Make an Experiment to find the optimal (flat) image plane location with 30 arcsec FOV
# This is not a fully defined problem, so I just minimize the rms of a set of rays that all come in at 30 arcsec off of
# parallel
exp_5 = experiment.Experiment()

# bring in the instrument
exp_5.add_instrument(ii.cass)

# bring the 30 arcsecond rays
rays_5 = ri.five_min_rays
exp_5.set_ray_starts(rays_5)
nominal_results = exp_5.run()
f_num_2 = ii.cass_inputs.f_num_total - ii.cass_inputs.f_num_1
suggested_offset = f_num_2 * nominal_results.mean_spread    # as a first guess, the offset required should be related to
                                                            # the spread with no offset and the f/# of the secondary
suggested_center = 3.15 - suggested_offset  # 3.15 is the paraxial focus
explore_space = np.arange(suggested_center, 3.15, 1E-6)  # the assumption here is that I will only have the capability
                                                         # to adjust my instrument with 1 micron precision, so why
                                                         # search any finer grid than that?
rms_list = []
for pt in explore_space:
    exp_5.instrument.detector.L_r_L[0][0] = pt
    exp_5.reset()
    rms_list.append(exp_5.run().mean_spread)

min_rms_i = np.argmin(rms_list)
min_rms_at  = explore_space[min_rms_i]
print("minimum rms at : ", min_rms_at)
print("an offset of: ", min_rms_at - 3.15)

if make_plots:
    fig = plt.figure()
    ax = fig.add_subplot('111')
    ax.set_title('rms vs position')
    ax.plot((explore_space - 3.15) * 1E6, np.array(rms_list)*1E6)
    ax.set_ylabel('image RMS [um]')
    ax.set_xlabel('position relative to nominal [um]')
    if save_plots:
        plt.savefig('./figures/off5/off5.png')
    if show_plots:
        plt.show()
    plt.close('all')



