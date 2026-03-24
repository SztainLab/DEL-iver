<img src="https://github.com/SztainLab/DEL-iver/blob/main/logo.png" width="200"/>

Package for processing DNA-encoded library data, training ML models, and picking hits from make on demand libraries

Order to run scripts:
  - ECFP4calculator.py
  - Make_BBdictionaries.py
  - Split_TestTrainVal.py
  - Run_Allmodels.py


## For development 
`pip install -e .` 

## Basic usage 
```
import DEL_iver as DEL



print("=="*20,"Testing with full data loaded at once including misc and specified id column")
# passing explicit id colum, and including bonus columns using misc
data=DEL.Data_Reader.from_csv('DEL_iver/tests/data/example_messy_input.csv',
    building_blocks=["col_2", "col_3", "col_5"],  # list of building block columns
    molecule_smiles="col_6",
    id_col="col_4",
    misc_cols=["col_1"]
)

print(data.columns)
print(data.head(n=2))
train_df, val_df, test_df=DEL.split_data(data)
print(len(train_df),len(val_df),len(test_df))
# 1599 40 360

print("=="*20,"Testing with iterator from csv, without including misc or specified id column")
#Generates own sequential id column if not provided, and does not includes misc columns
data=DEL.Data_Reader.from_csv('DEL_iver/tests/data/example_messy_input.csv',
    building_blocks=["col_2", "col_3", "col_5"],  # list of building block columns
    molecule_smiles="col_6",
)

print(data.columns)
print(data.head(n=2))
train_df, val_df, test_df=DEL.split_data(data)
print(len(train_df),len(val_df),len(test_df))
# 1599 40 360

print("=="*20,"Testing with iterator from csv")
data_iterator=DEL.Data_Reader.iterator_from_csv('DEL_iver/tests/data/example_messy_input.csv',
                                                chunk_size=500,
                                                building_blocks=["col_2", "col_3", "col_5"],
                                                molecule_smiles="col_6"
                                                )
#print(type(data))
train_df, val_df, test_df=DEL.split_data(data_iterator)
print(len(train_df),len(val_df),len(test_df))
# 1599 40 360



```

Machine learning model descriptions and usage for models in `models.py`:
  - ...
  - ...
