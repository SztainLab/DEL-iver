<img src="https://github.com/SztainLab/DEL-iver/blob/main/logo.png" width="200"/>

Package for processing DNA-encoded library data, training ML models, and picking hits from make on demand libraries

# Quick Start for Beginners

The defult functions can all be carried out using a single python script after installation following the steps below:

## 1. Copy this github repository to your local device.

   * This currently requires a linux environment, which is standard for Mac. For windows users, we reccomend first installing [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/windows-gui-install).

In your terminal, type

   ```
   git clone https://github.com/SztainLab/DEL-iver.git
   ```

## 2. Install your local copy 
```
pip install -e . 
```
## 3. Prepare input data

The input must be a CSV file where:

Each row represents a single compound.

Each column represents a building block.

Each building block column contains a valid SMILES string.

You must explicitly provide the building block column names when loading the data. A label column contain either 0 or 1 can also be used for computing Pbind.

Example format:

| bb1_smiles | bb2_smiles | bb3_smiles |binds|
| ---------- | ---------- | ---------- | ----|
| CCCO       | c1ccccc1   | CCN        | 1   |



4. run!

```
python DEL_iver_results.py
```
   
What this script does:

1. Reads DEL data ...
2. Convert it to storage / memory efficient parqut format
3. enumerates all building blocks
4. computes metrics for pbind
5. helps filter to find the top molecules per pbind
6. Generates ECFP4 embeddings for each building block
7. trains models on the fingerprints
8. Generates plots at several steps along the process for visualization


## Troubleshooting 
TBD

## Advanced Settings
TBD
