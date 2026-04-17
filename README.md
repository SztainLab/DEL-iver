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

- Each row represents a single compound.

- Each column represents a building block or data relating to it.

- Each building block column contains a valid SMILES string.

You must explicitly provide the building block column names when loading the data. A label column contain either 0 or 1 can also be used for computing Pbind.

Example format:

| bb1_smiles | bb2_smiles | bb3_smiles |binds|
| ---------- | ---------- | ---------- | ----|
| CCCO       | c1ccccc1   | CCN        | 1   |



## 4. Set variables and execute

At the top of the `DEL_iver_results.py` script, only a small number of configuration variables need to be defined before running the analysis.

---

### Required configuration

```python id="cfg1"
input = "path/to/input.csv"
```

Path to the input CSV file containing the DEL dataset.

---

```python id="cfg2"
bb_cols = [
    "buildingblock1_smiles",
    "buildingblock2_smiles",
    "buildingblock3_smiles"
]
```

List of building block columns in the dataset.

* The **order of this list is important**
* It defines the mapping:

  * `bb_cols[0]` → BB1
  * `bb_cols[1]` → BB2
  * `bb_cols[2]` → BB3

Each column must contain valid SMILES strings.

---

```python id="cfg3"
label = "binds"
```

Binary label column (optional or required depending on the selected metric):

* Must contain only:

  * `1` → hit / binder
  * `0` → non-hit / non-binder

This column is used for enrichment and performance-based metrics.

---

### Optional configuration

```python id="cfg4"
output = "path/to/output"
```

Output directory where results, figures, and computed tables will be saved, if not specified results get written to the file systems cache directory.

---





### Run the pipeline

Once the variables are set, execute the full analysis pipeline with:
```
python DEL_iver_results.py
```
What this script does:

1. Reads DEL data and convert it to storage / memory efficient parquet format.
3. Enumerates all building blocks and disynthon combinations possible within the data. 
4. Computes pbind.
5. Filters to find the top molecules per pbind.
6. Generates ECFP4 embeddings for each building block
7. Trains models on the fingerprints
8. Generates plots at several steps along the process for visualization


## Troubleshooting 
TBD

## Advanced Settings
TBD
