#!/bin/python

import DEL_iver as deliv

# provide input and instantiate ddr class
# input="" You will need to provide your own input csv here. it should be the same as the input csv you provide to del_iver_results.py
# the following is an example csv found in the data directory
input="data/example.csv"
bb_cols=["bb1_smiles","bb2_smiles","bb3_smiles"]
output_prefix ='example_prefix'

ddr = deliv.DataReader.from_csv(input,building_blocks=bb_cols, label='binds')

# generate the bb id to SMILES dictionaries 
table, id_to_smile = deliv.generate_bb_dictionaries(ddr)

# generate ECPF4 fingerprints from SMIILES
deliv.gen_fingerprints(ddr, output_prefix=output_prefix)

# train the default model on BB ECFP4 fingerprints
deliv.train_default(ddr, output_prefix=output_prefix)

# inference on the test file created while training
deliv.inference(ddr, output_prefix=output_prefix)

# uncomment the following if you'd like to train an invariant model and inference using that model
# deliv.train_invariant(ddr, output_prefix='testing')
# deliv.inference(ddr, output_prefix='testing')