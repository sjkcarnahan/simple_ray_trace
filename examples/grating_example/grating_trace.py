'''
Scott Carnahan
Experiment - Cassegrain with Rowland Circle Grating
'''

from srt_modules import experiment as rt
from srt_instances import instrument_instances as ii, ray_instances as ri
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate  # note, you can pip install this. It's a handy little package that displays or saves
                                # tables nicely.

exp = rt.Experiment()
rays = ri.basic_paraxial_rays
exp.set_ray_starts(rays.X)
exp.set_ray_start_dir(rays.d)

exp.add_instrument(ii.grating_cassegrain)
grating = ii.grating  # grab for convenience
detector = ii.cylindrical_detector  # grab for convenience
wavelength_list = ri.wavelength_list

data_list = []  # each entry will be the scatter plot from a single wavelength, single diffraction order
for order in [0, 1, 2]:
    grating.set_order(order)
    for wavelength in wavelength_list:
        grating.set_wavelength(wavelength)
        exp.reset()
        exp.trace_rays()
        data_list.append(detector.extract_image())
    if order == 1:
        # this is to be done only on the 1st order trace
        # this gives the spectral resolving power of the grating/instrument at a few wavelengths and prints them
        angstrom_per_mm = 1E7 / 3600. / 1000.
        x1600 = data_list[-5][0, :]
        dx_1600 = (np.nanmax(x1600) - np.nanmin(x1600)) * 1000.
        resolution_1600 = dx_1600 * angstrom_per_mm
        resolving_power_1600 = 1600. / resolution_1600
        print('resolving power at 1600 A: %f' % resolving_power_1600)
        x1200 = data_list[-9][0, :]
        dx_1200 = (np.nanmax(x1200) - np.nanmin(x1200)) * 1000.
        resolution_1200 = dx_1200 * angstrom_per_mm
        resolving_power_1200 = 1200. / resolution_1200
        print('resolving power at 1200 A: %f' % resolving_power_1200)
        x2000 = data_list[-1][0, :]
        dx_2000 = (np.nanmax(x2000) - np.nanmin(x2000)) * 1000.
        resolution_2000 = dx_2000 * angstrom_per_mm
        resolving_power_2000 = 2000. / resolution_2000
        print('resolving power at 2000 A: %f' % resolving_power_2000)
        headers = ['Wavelength', 'Resolving Power']
        data = np.array([resolving_power_1200, resolving_power_1600, resolving_power_2000]).reshape([3, 1])
        index = np.array([1200, 1600, 2000]).reshape([3, 1])
        dat = np.hstack([index, data])
        table = tabulate(dat, headers, tablefmt='latex')  # tabulate really shines in saving data for tex reports
        with open('./resolvingTable.txt', 'w') as f:
            f.write(table)
            f.close()
        # fig1 = rt.save_3d_plot(inst.surfaces, exp.ray_hist)  # This is useful for debugging but ugly. The detector is un-
        # realistically large to be able to catch 0th-2nd orders all at once.
        # plt.savefig('./figures/lab_view.png')

scat_fig = plt.figure(figsize=(20, 5))

ax2 = scat_fig.add_subplot('111')
for order in range(3):
    for col, wavelength in zip(ri.colors, data_list):
        ax2.scatter(wavelength[0, :], wavelength[1, :]*100, s=1, color=col)
ax2.set_xlabel('RA [rad]')
ax2.set_ylabel('height (along cylinder axis) [cm]')
ax2.legend([str(int(w)) for w in wavelength_list], markerscale=6, title='[A]')
ax2.text(0.575, 2., '0 order', bbox=dict(facecolor='black', alpha=0.05))
ax2.text(0., 2., 'first order', bbox=dict(facecolor='black', alpha=0.05))
ax2.text(-0.6, 2., 'second order', bbox=dict(facecolor='black', alpha=0.05))
plt.savefig('./figures/spectrum.png')
plt.show()



