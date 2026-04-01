<img src="https://github.com/SztainLab/DEL-iver/blob/main/logo.png" width="200"/>

Package for processing DNA-encoded library data, training ML models, and picking hits from make on demand libraries

Order to run scripts:
  - ECFP4calculator.py
  - Make_BBdictionaries.py
  - Split_TestTrainVal.py
  - Run_Allmodels.py


## For development 

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


## Basic usage 
```

import DEL_iver as deliv

print("=="*20,"Testing with chunks from csv")
ddr=deliv.Data_Reader.from_csv('DEL_iver/tests/data/example_messy_input.csv',
                                                chunk_size=500,
                                                building_blocks=["col_2", "col_3", "col_5"],
                                                molecule_smiles="col_6",
                                                misc_cols=["col_1"]
                                                )
print(type(ddr))
print(vars(ddr))

print("BUILDING BLOCK FALSE")
full_molecule_smile_to_int, full_molecule_int_to_smile=deliv.generate_BB_dictionaries(ddr,bb_fingerprints=False)


# --- Molecules ---
print(f"Full molecule dict (SMILES -> int), total entries: {len(full_molecule_smile_to_int)}")
print("Preview:", list(full_molecule_smile_to_int.items())[:2])

print(f"Full molecule dict (int -> SMILES), total entries: {len(full_molecule_int_to_smile)}")
print("Preview:", list(full_molecule_int_to_smile.items())[:2])


full_molecule_smile_to_int, full_molecule_int_to_smile,bb_smile_to_int, bb_int_to_smile=deliv.generate_BB_dictionaries(ddr,bb_fingerprints=True)



print("BUILDING BLOCK TRUE")
# --- Building blocks ---
for colname, mapping in bb_smile_to_int.items():
    print(f"\nBuilding block column '{colname}' (SMILES -> int), total entries: {len(mapping)}")
    print("Preview:", list(mapping.items())[:2])

for colname, mapping in bb_int_to_smile.items():
    print(f"\nBuilding block column '{colname}' (int -> SMILES), total entries: {len(mapping)}")
    print("Preview:", list(mapping.items())[:2])




```

Machine learning model descriptions and usage for models in `models.py`:
  - ...
  - ...
