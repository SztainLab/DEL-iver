#!/bin/python

import DEL_iver as deliv

enamine_db = 'data/enamine_bbs_small.csv'
output_prefix ='example_prefix'

# calculate ECFP4 fingerprints of analog molecules, calculate analog similarities
# propose analogs for each DEL molecule
deliv.analog_embed(ddr, enamine_db, output_prefix=output_prefix)

# predict binding probability (from the models trained in del_iver_models.py) of
# proposed analogs
deliv.inference_analog_moles(ddr, output_prefix=output_prefix)