#!/bin/python

import DEL_iver as deliv

# calculate ECFP4 fingerprints of analog molecules, calculate analog similarities
# propose analogs for each DEL molecule
# the same output_prefic as the above functions should be used here
deliv.analog_embed(ddr, 'data/enamine_bbs_small.csv', output_prefix='testing')

# predict binding probability (from the models trained in del_iver_models.py) of
# proposed analogs
# the same output_prefix as output_prefix used to train models should be used here
deliv.inference_analog_moles(ddr, output_prefix='testing')