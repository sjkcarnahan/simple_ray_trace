'''
Scott Carnahan
Experiment - Various Angles
Spring 2019
There is no goal here other than showing the aberrations on the focal plane due to non-paraxial rays with various
incoming angles. Coma should be quite evident.
'''

import matplotlib.pyplot as plt
from modules import rayTracing as rt
from instances import project_inputs as pi

# some figure flags
show_plots = True
save_plots = True

# make an Experiment
exp_2 = rt.Experiment()

# bring in the instrument
exp_2.add_instrument(pi.cass)

# get the rays to trace
ray_sets = pi.angled_ray_list
angle_set = pi.various_angles


# run the Experiments
result_plot_list = []
for angle, ray in zip(angle_set, ray_sets):
    exp_2.name = str(angle)
    exp_2.set_ray_starts(ray.X)
    exp_2.set_ray_start_dir(ray.d)
    exp_2.run()
    result_plot_list.append(exp_2.result_plot())

# plotting (plots actually generated in the loop above
if save_plots:
    for fig, ang in zip(result_plot_list, angle_set):
        plt.figure(fig.number)
        plt.savefig('./figures/various_angles/' + str(int(ang)) + '.png')
if show_plots:
    plt.show()

plt.close('all')


