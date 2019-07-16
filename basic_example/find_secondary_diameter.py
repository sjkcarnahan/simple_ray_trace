'''
Scott Carnahan
ASTR 5760 Proj 1 - find secondary diameter
Spring 2019
'''
import sys
sys.path.append('../modules/')

import rayTracing as rt
from instances import project_inputs as pi

# define the Experiment to find the necessary secondary diameter
exp_diam = rt.Experiment()

# bring in the instrument
exp_diam.add_instrument(pi.cass)

# bring in the one off axis edge ray
edge_ray = pi.edge_ray
exp_diam.set_ray_starts(edge_ray.X)
exp_diam.set_ray_start_dir(edge_ray.d)

# do it to it
exp_diam.run()
nec_diam = exp_diam.ray_hist[3][1]*2

with open('./textOutput/necDiam.txt', 'w') as f:
    f.write(str(nec_diam[0]))