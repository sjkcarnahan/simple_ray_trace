'''
Scott Carnahan
For more descriptive comments, please see find_image_plane_30.py
'''

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from srt_modules import experiment as rt
from srt_instances import instrument_instances as ii, ray_instances as ri

# Make an Experiment to find the optimal (flat) image plane location with 30 arcsec FOV
exp_30 = rt.Experiment()

# bring in the instrument
exp_30.add_instrument(deepcopy(ii.cass))

# bring the 30 arcsecond rays
rays_30 = deepcopy(ri.thirty_sec_rays)
exp_30.set_ray_starts(rays_30)
nominal_results = exp_30.run()
f_num_2 = ii.cass_inputs.f_num_total - ii.cass_inputs.f_num_1
suggested_offset = f_num_2 * nominal_results.mean_spread
suggested_center = 3.15 - suggested_offset
delta = suggested_offset * .1
explore_space = np.arange(suggested_center, 3.15, 1E-6)
rms_list = []
for pt in explore_space:
    exp_30.instrument.detector.L_r_L[0][0] = pt
    exp_30.reset()
    rms_list.append(exp_30.run().mean_spread)

min_rms_i = np.argmin(rms_list)
min_rms_at  = explore_space[min_rms_i]

def test_offset():
    assert np.fabs((min_rms_at)/3.1499963816399332) - 1.0 < 1E-14



