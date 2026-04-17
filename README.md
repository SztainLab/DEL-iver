<img src="https://github.com/SztainLab/DEL-iver/blob/main/logo.png" width="200"/>

Package for processing DNA-encoded library data, training ML models, and picking hits from make on demand libraries

## Quick Start for Beginners

The defult functions can all be carried out using a single python script after installation following the steps below:

1. Copy this github repository to your local device.

   * This currently requires a linux environment, which is standard for Mac. For windows users, we reccomend first installing [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/windows-gui-install).

In your terminal, type

   ```
   git clone https://github.com/SztainLab/DEL-iver.git
   ```
2. run setup ...
3. ensure data is in the correct format and folder ...
4. run!

```
python DEL_iver_results.py
```
   
What this script does:

1. Reads DEL data ...
2. ...

Output files: 

1. results ... 

## Troubleshooting 

## Advanced Settings




## For development 

Order to run scripts:
  - ECFP4calculator.py
  - Make_BBdictionaries.py
  - Split_TestTrainVal.py
  - Run_Allmodels.py

To install when in current directory:
`pip install -e .` 

the heart of the code is the data reader, it contains everything that is needed to run a calculation on the source data, as of Mar/25 it contains the following attributes


<class 'DEL_iver.data_loader.data_reader.Data_Reader'>
'building_blocks': ['col_2', 'col_3', 'col_5'],  #list of user specified columns that contain building blocks 

'molecule_smiles': 'col_6', #column with entire molecule smile

'misc_cols': ['col_1'], #a list of all other columns to include in the data loader

'n_building_blocks': 3, #this gets computed from user inputs, its the len of the building block list

'source_file': 'DEL_iver/tests/data/example_messy_input.csv', #this is the original source file that got read in to instantiate the class, it does not stay read in memory untill a function is called on it

'chunk_size': 500, #user specified amount of rows to have in each chunk of data 

'n_chunks': 4, #this gets computed by the class it self, tell you how many chunk_size created

 'actual_chunk_sizes': [500, 500, 500, 499] #list of the size of each chunk gets computed by the class


## Analysis usage 
```

import DEL_iver as deliv

ddr=deliv.DataReader.from_csv('/users/group/DEL-iver/DEL_iver/tests/data/large_example.csv',
                                                memory_per_chunk_mb=300,
                                                building_blocks=["buildingblock1_smiles", "buildingblock2_smiles", "buildingblock3_smiles"]

                                                )

print(vars(ddr))
print(ddr.data)
print(type(ddr.data))
print(ddr.get_chunk(10))
print(type(ddr.get_chunk(10)))
print(ddr.n_chunks)
deliv.split_data(ddr)


#this will split the data, it returns a table with a single column wit 0,1,2 as identifying numbers that assigned it to a split, it also supports different splits, this also gets written to cache

splits=deliv.split_data(ddr,seed=123)

#table is per row the idenityfying number assigned to each smile, id_to_smile is a dictionary for loop up later
table, id_to_smile=deliv.generate_bb_dictionaries(ddr) 

#here the split_group variable was used which returns a dictionary PER split, if not it returns for entire data set also gets cachced 
bb_stats, disynthon_stats=deliv.compute_pbind_and_enrichment(ddr,method="epsilon",split_col="split_group")


```

Note on difference between pbind and enrichment: 

Pbind
This is the raw "Success Rate." It measures how often a building block appeared in the successful "Hit" pool relative to how many times it was present in the total library.

Enrichment: 
This is a "Signal-to-Noise" ratio. It compares the frequency of a building block in the Hits to its frequency in the Non-Hits (or the background library).



Machine learning model descriptions and usage for models in `models.py`:
  - ...
  - ...
