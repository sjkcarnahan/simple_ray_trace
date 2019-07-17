from srt_modules import experiment, light_sources, optical_surfaces, instruments
import numpy as np

exp = experiment.Experiment()
rayX, rayd = light_sources.make_one_edge_ray(0., 0.)
prim = optical_surfaces.ParabolicMirrorWithHole(0.2, 1.0, 0.1, [-np.pi/2., 0., 0.], [3., 0., 0.])
inst = instruments.Instrument()
inst.set_surfaces([prim])
exp.add_instrument(inst)
exp.set_ray_starts(rayX)
exp.set_ray_start_dir(rayd)
exp.trace_rays_test()
