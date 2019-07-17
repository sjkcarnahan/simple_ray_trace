'''
Scott Carnahan
simple ray trace tests.
If you're using pytest, be sure to import instances with deepcopy. Otherwise changes made to imported instances carry
over to other tests.
'''

from srt_modules import experiment as rt
from srt_instances import instrument_instances as ii, ray_instances as ri
import numpy as np
from copy import deepcopy

def test_basic_trace():
    # create an Experiment
    exp_1 = rt.Experiment()  # a container to hold the instrument and results etc. and run tests
    exp_1.name = "basic_trace"  # this can be used in file names when saving things

    # bring in the instrument
    exp_1.add_instrument(deepcopy(ii.cass))  # the instrument is defined in a different file to keep it simple and clean here

    # Make rays for this Experiment. In rendering lingo this is defining a light source, or multiple point light sources
    exp_1.set_ray_starts(deepcopy(ri.basic_paraxial_rays))

    # run and plot
    exp_1.reset()
    results = exp_1.run()  # steps through all surfaces to find ray intersections and reflections

    # all points should be co-located to within machine precision
    assert np.nanmax(results.image) < 1E-13

