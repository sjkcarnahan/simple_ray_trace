import sys
sys.path.append('../modules/')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import rayTracing as rt
import project_inputs as pi


show_plots = True
save_plots = True
make_plots = show_plots or save_plots

# Make an Experiment to find the optimal (flat) image plane location with 30 arcsec FOV
exp_5 = rt.Experiment()

# bring in the instrument
exp_5.add_instrument(pi.cass)

# bring the 30 arcsecond rays
rays_5 = pi.five_min_rays
exp_5.set_ray_start_dir(rays_5.d)
exp_5.set_ray_starts(rays_5.X)
nominal_results = exp_5.run()
f_num_2 = pi.cass_inputs.f_num_total - pi.cass_inputs.f_num_1
suggested_offset = f_num_2 * nominal_results.mean_spread
suggested_center = 3.15 - suggested_offset
delta = suggested_offset * .1
explore_space = np.arange(suggested_center, 3.15, 1E-6)
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



