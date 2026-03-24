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


data=DEL.Data_Reader.from_csv('DEL_iver/tests/data/example.csv')
print(len(data))
#1999

train_df, val_df, test_df=DEL.split_data(data)
print(len(train_df),len(val_df),len(test_df))
# 1599 40 360
```

Machine learning model descriptions and usage for models in `models.py`:
  - ...
  - ...
