'''
Scott Carnahan
Experiment - Test Cassegrain with Rowland Circle Grating
'''

from srt_modules import experiment as rt
from srt_instances import instrument_instances as ii, ray_instances as ri
import numpy as np
from copy import deepcopy

exp = rt.Experiment()
rays = deepcopy(ri.basic_paraxial_rays)
exp.set_ray_starts(rays.X)
exp.set_ray_start_dir(rays.d)

exp.add_instrument(deepcopy(ii.grating_cassegrain))
grating = deepcopy(ii.grating)  # grab for convenience
detector = deepcopy(ii.cylindrical_detector)  # grab for convenience


grating.set_order(1)
grating.set_wavelength(1200.)
exp.reset()
exp.trace_rays()

angstrom_per_mm = 1E7 / 3600. / 1000.
x1200 = detector.extract_image(exp.ray_hist[-1])[0, :]
dx_1200 = (np.nanmax(x1200) - np.nanmin(x1200)) * 1000.
resolution_1200 = dx_1200 * angstrom_per_mm
resolving_power_1200 = 1200. / resolution_1200

def test_grating_trace():
    assert np.fabs(resolving_power_1200 / 679.708744) - 1 < 1E-13



