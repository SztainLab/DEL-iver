<p align="center">
  <img src="https://github.com/SztainLab/DEL-iver/raw/main/logo.png" width="200"/>
</p>

Package for processing DNA-encoded library data, training ML models, and picking hits from make on demand libraries

Citation: Dolorfino, M.; Perez, D. S.; Fu, Y.; Lin, S.-H.; McCarty, S.; O’Meara, M. J.; Sztain, T. Assessing the Generalizability of Machine Learning and Physics Methods for DNA-Encoded Libraries. <i>bioRxiv</i> April 19, 2026 [link](https://doi.org/10.64898/2026.04.18.719394)


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





### Run the results pipeline

Once the variables are set, execute the full analysis pipeline with:
```
python DEL_iver_results.py
```
What this script does:

1. Reads DEL data and convert it to storage / memory efficient parquet format.
2. Enumerates all building blocks and disynthon combinations possible within the data. 
3. Computes pbind.
4. Filters to find the top molecules per pbind.
5. Generates figures of building block and disynthon distributions


It outputs: 
1) Building block (BB) distribution plots (bbs.png)
2) Disynthon distribution plots (disynthons.png)
3) Top 10 BB and structures from data (bb_structures.png)
4) Top 10 disynthon and structures from data(disynthon_structures.png)
5) Table of disynthon statistics 

**Note that for the `DEL_iver_models.py` and `DEL_iver_analogs.py` scripts, the output_prefix must be set to the same string**
### Run the model training pipeline

The variables you will need to set at the top of the `DEL_iver_models.py` are the input file (which is the same as the one you provided for the `DEL_iver_results.py` script
and the output_prefix.

Once the variables are set, execute the full model training/inference pipeline with:
```
python DEL_iver_models.py
```

What this script does: 
1. Reads DEL data and convert it to storage / memory efficient parquet format.
2. Enumerates all building blocks and disynthon combinations possible within the data.
3. Computes the ECFP4 fingerprints of all building blocks in the DEL dataset.
4. Splits the DEL dataset in 80/20 train/test split.
5. Trains the default building block ML model on the train split.
6. Performs inference using the trained model on all of the molecules in the test dataset.
7. If the script writes out a file or png, it will state where that file has been written. 

It outputs:
1) Parquet files of the BB ECFP4 fingerprints
2) Parquet files of the train and test datasets that result from the 80/20 train/test split
3) The trained model as a `.pth` file
4) A parquet file with the predicted binding probabilities of the model on the test set
5) AUROC plot showing the performance of the trained model on the test set (png)
6) Precision recall plot showing the performance of the trained model on the test set (png)

### Run the analog proposal pipeline

The variables you need to at the top of the `DEL_iver_analogs.py` are the path to the analog csv that you want to analyze (the only requirment here is that there is a `SMILES` column) and the output_predix.

Once the variables are set, execute the full analog similarity comparison / analog proposal pipeline with:
```
python DEL_iver_analogs.py
```

What this script does:
1. Reads the analog dataset provided.
2. Computes ECFP4 fingerprints of the analog molecules.
3. Generates a UMAP embedding of the DEL bb molecules and the analog molecules.
4. Calculates pairwise tanimoto similarity scores between all DEL bbs against all analog molecules.
5. For each DEL molecule, which consists of 3 bbs, an analog for each bb is chosen as molecule with highest similarity to the specific bb.
   For example, if there is DEL bb1, bb2, bb3, there will be analog1, analog2, analog3, one for each bb that makes up the full DEL molecule.
6. Performs inference on each of the sets of proposed analogs per one DEL molecule. A binding probability is predicted.
7. If the script writes out a file or png, it will state where that file has been written. 

It outputs:
1) A parquet file of the ECFP4 fingerprints for the analog dataset provided
2) A parquet file of the UMAP embedding coordinates for each bb and analog model, with labels identifying which set the molecule belongs to
3) A png of the UMAP embedding, colored by the different molecule sets
4) A parquet file containing the proposed analogs and the tanimoto similarity scores of the proposed analogs
5) A parquet file that contains the predicted binding probabilities of the proposed analogs

## Troubleshooting 
TBD

## Advanced Settings
TBD
