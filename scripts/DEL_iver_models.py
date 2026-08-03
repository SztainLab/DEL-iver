#!/bin/python

import DEL_iver as deliv

# provide input and instantiate ddr class
# input="" You will need to provide your own input csv here. it should be the same as the input csv you provide to del_iver_results.py
# the following is an example csv found in the data directory
input="data/example.csv"
bb_cols=["bb1_smiles","bb2_smiles","bb3_smiles"]
output_prefix ='example_prefix'
label='binds' # the label can be binary or continuous (IC50, readcounts, etc.) # specify column name of labels here
continuous_label = False # train only on binary labels. If you want to train on continuous values, change to True
full_molecule_smiles = None # if you want to do analysis with full molecules, specify the column name containing fullmole smiles here

# default for 
ddr = deliv.DataReader.from_csv(input, building_blocks=bb_cols, label=label, molecule_smiles=full_molecule_smiles)

# generate the bb id to SMILES dictionaries
deliv.run_enumeration(ddr)

# generate ECPF4 fingerprints from SMIILES
deliv.gen_fingerprints(ddr, output_prefix=output_prefix, fullmole=False)

# train the default model on BB ECFP4 fingerprints
deliv.train_default(ddr, output_prefix=output_prefix, continuous_label=continuous_label)

# inference on the test file created while training
deliv.inference_default(ddr, output_prefix=output_prefix)

# uncomment the following if you'd like to train an invariant model on BB ECPF4 fingerprints and inference using that model
# deliv.train_invariant(ddr, output_prefix='invariant', continous_label=continuous_label)
# deliv.inference_invariant(ddr, output_prefix='invariant')

# uncomment the following if you'd like to train a full molecule model and inference using that model
# deliv.train_fullmole(ddr, output_prefix='fullmolecule', continuous_label=continuous_label)
# deliv.inference_fullmole(ddr, output_prefix='fullmolecule', continuous_label=continuous_label)