#!/bin/python

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import sys 
import pickle
import os
from itertools import combinations
import itertools
from DEL_iver.utils.utils import *
from DEL_iver.utils.cache import CacheManager, CacheNames
import umap
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'font.family': 'sans-serif',
         'legend.fontsize': '12',
         'figure.figsize': (12,12),
         'axes.labelsize': '24',
         'axes.titlesize': '24',
         'xtick.labelsize': '24',
         'ytick.labelsize': '24'}
pylab.rcParams.update(params)
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def compute_labeled_similarity(ecfp4_list1, ecfp4_list2, smiles_list1, smiles_list2, labels1=None, labels2=None):
    """
    Compute pairwise similarities with proper label tracking.
    
    Returns:
        matrix: 2D array of similarity scores
        row_labels: Labels for list1 (uses indices if none provided)
        col_labels: Labels for list2 (uses SMILES if none provided)
    """
    if labels1 is None:
        labels1 = smiles_list1
    if labels2 is None:
        labels2 = smiles_list2  
    
    fps1 = ecfp4_list1
    fps2 = ecfp4_list2
    
    def ensure_rdkit_bitvect(fp):
        if fp is None:
            return None
        # If it's a numpy array, convert it back to ExplicitBitVect
        if isinstance(fp, np.ndarray):
            bitvect = DataStructs.ExplicitBitVect(len(fp))
            for i, bit in enumerate(fp):
                if bit:
                    bitvect.SetBit(i)
            return bitvect
        return fp

    fps1 = [ensure_rdkit_bitvect(fp) for fp in fps1]
    fps2 = [ensure_rdkit_bitvect(fp) for fp in fps2]
    
    matrix = np.full((len(fps1), len(fps2)), np.nan)
    
    for i, fp1 in enumerate(fps1):
        if fp1 is not None:
            valid_indices = [j for j, fp2 in enumerate(fps2) if fp2 is not None]
            valid_fps = [fps2[j] for j in valid_indices]
            
            if valid_fps:
                scores = DataStructs.BulkTanimotoSimilarity(fp1, valid_fps)
                for j, score in zip(valid_indices, scores):
                    matrix[i, j] = score
    
    return matrix, labels1, labels2

def get_best_matches(matrix, labels1, labels2):
    """
    For each item in labels1, find the best match in labels2 based on similarity matrix.
    
    Returns:
        dict: {labels1[i]: [best_labels2, best_score]} for each i
    """
    best_matches = {}
    
    for i, label1 in enumerate(labels1):
        # Get similarity scores for this query
        scores = matrix[i, :]
        
        # Skip if all scores are NaN (e.g., invalid fingerprint)
        if np.all(np.isnan(scores)):
            best_matches[label1] = [None, np.nan]
            continue
        
        # Find index of maximum similarity
        best_idx = np.nanargmax(scores)
        best_score = scores[best_idx]
        best_label2 = labels2[best_idx]
        
        best_matches[label1] = [best_label2, best_score]
    
    return best_matches

def add_best_match_columns(df, smiles_column, best_matches_dict, analog_col_name, tanimoto_col_name):
    """
    Add analog and tanimoto columns to dataframe based on best matches dictionary.
    
    Returns:
    --------
    df : pandas.DataFrame
        Dataframe with new columns added
    """
    # Create mapping series from dictionary
    analog_map = {k: v[0] for k, v in best_matches_dict.items()}
    tanimoto_map = {k: v[1] for k, v in best_matches_dict.items()}
    
    # Map values to new columns
    df[analog_col_name] = df[smiles_column].map(analog_map)
    df[tanimoto_col_name] = df[smiles_column].map(tanimoto_map)
    
    return df

def analog_embed(ddr, enamine_input, output_prefix, ecfp4_size=1024):
    """
    Calculate ECFP4 fingerprints of Enamine building blocks and embed with DEL building blocks.
    Produce UMAP embedding parquet files and plots of the embeddings.
    Compute tanimoto scores between DEL bbs and Enamine building blocks, and propose the 
    most similar analog for all molecules in the DEL dataset. 
    
    Parameters:
    -----------
    enamine_input: 
        Path to a csv file containing SMILES strings of analogs to be embedded with DEL bbs.
        The file can contain any number of columns, but it MUST contain a 'SMILES' column that contains
        the SMILES strings of analogs. 
    output_prefix : str
        A string specifying the prefix for output files. 
        E.G if output_prefix='analogs', the output would be named 'analog_fingerprints.parquet'
    ecfp4_size : int, default=1024
        Size of ECFP4 fingerprints
        
    Returns:
    --------
    parquet file 
        parquet file containing the ECFP4 fingerprints of analogs
        parquet file containing the UMAP embedding of the analogs with the DEL bbs
        output a parqet file containing the DEL dataset and proposed analogs (bb1, bb2, bb3) for each DEL molecule
    png
        png of plots of each of the UMAP embeddings listed above
    
    Example:
    --------
    >>> analog_embed(
            ddr,
    ...     output_prefix="experiment1",
    ...     chunk_size=1000000,
    ...     ecfp4_size=2048
    ... )
    """
    
    print(f'Using {multiprocessing.cpu_count()} CPUs...')
    print(f'Output prefix: {output_prefix}')
    print(f'ECFP4 size: {ecfp4_size}')
    
    source_file = ddr.source_file
    parquet_source = pq.ParquetFile(source_file)
    df_source = parquet_source.read().to_pandas()
    
    # print(df_source.head())
    # print(df_source.columns)
    # print(len(df_source))
    
    # load the bb1,bb2,bb3 DEL ecfp4 dicts
    # get the bb fingerprint parquets
    bb1s = ddr.cache.get_path(
        CacheNames.SMILESEMBEDDING,
        filename=f"{output_prefix}_bb1_fingerprints.parquet"
    )

    # get the bb fingerprint parquets
    bb2s = ddr.cache.get_path(
        CacheNames.SMILESEMBEDDING,
        filename=f"{output_prefix}_bb2_fingerprints.parquet"
    )

    # get the bb fingerprint parquets
    bb3s = ddr.cache.get_path(
        CacheNames.SMILESEMBEDDING,
        filename=f"{output_prefix}_bb3_fingerprints.parquet"
    )

    parquet_bb1 = pq.ParquetFile(bb1s)
    df_bb1 = parquet_bb1.read().to_pandas()
    bb1_dict = dict(zip(df_bb1.iloc[:, 0], df_bb1.iloc[:, 1]))
    del df_bb1

    parquet_bb2 = pq.ParquetFile(bb2s)
    df_bb2 = parquet_bb2.read().to_pandas()
    bb2_dict = dict(zip(df_bb2.iloc[:, 0], df_bb2.iloc[:, 1]))
    del df_bb2

    parquet_bb3 = pq.ParquetFile(bb3s)
    df_bb3 = parquet_bb3.read().to_pandas()
    bb3_dict = dict(zip(df_bb3.iloc[:, 0], df_bb3.iloc[:, 1]))
    del df_bb3
    
    # load the positional ids of building blocks
    filename = ddr.cache.get_path(
        CacheNames.BB_DICTIONARIES,
        filename=f"{CacheNames.BB_DICTIONARIES.value}.{ddr.source_file.stem}.parquet"
    )
    parquet_alldata = pq.ParquetFile(filename)
    all_data_df = parquet_alldata.read().to_pandas()    
    
    all_data_df = all_data_df[all_data_df['buildingblock1_smiles_positional_id'].isin(list(bb1_dict.keys()))]
    all_data_df = all_data_df[all_data_df['buildingblock2_smiles_positional_id'].isin(list(bb2_dict.keys()))]
    all_data_df = all_data_df[all_data_df['buildingblock3_smiles_positional_id'].isin(list(bb3_dict.keys()))]
    
    # load the dict that has ids to smiles
    filename = ddr.cache.get_path(
    CacheNames.BB_DICTIONARIES,
    filename=f"{CacheNames.BB_DICTIONARIES.value}.{ddr.source_file.stem}_id_to_smiles.parquet"
    )
    
    parquet_sm = pq.ParquetFile(filename)
    df_id2smile = parquet_sm.read().to_pandas()
    id2smile = dict(zip(df_id2smile.iloc[:, 0], df_id2smile.iloc[:, 1]))
    del df_id2smile
    
    # load the enamine csv file
    enamine_df = pd.read_csv(enamine_input)
    enamine_smiles2ecfp4_dict = {}
    for smi in enamine_df['SMILES']:
        fp = retrieve_mol_fp(smi, 'ECFP4', ecfp4_size)
        enamine_smiles2ecfp4_dict[smi] = fp
        
    del enamine_df
    enamine_df = pd.DataFrame({'SMILES': enamine_smiles2ecfp4_dict.keys(), 'ECFP4s': enamine_smiles2ecfp4_dict.values()})
    
    # write enamine ecfp4 fingerprints to file
    output_out = ddr.cache.get_path(
        CacheNames.ANALOGS,
        filename=f"{output_prefix}_enaminefingerprints.parquet"
    )
    table = pa.Table.from_pandas(enamine_df)
    pq.write_table(table, (output_out))
    print(f"Enamine fingerprints saved to: {output_out}")
    
    bb1_dict = {id2smile[k]: v for k, v in bb1_dict.items()}
    bb2_dict = {id2smile[k]: v for k, v in bb2_dict.items()}
    bb3_dict = {id2smile[k]: v for k, v in bb3_dict.items()}
    
    # make umap embeddings 
    mol_labels = (['bb1'] * len(bb1_dict) + 
                ['bb2'] * len(bb2_dict) + 
                ['bb3'] * len(bb3_dict) + 
                ['analog'] * len(enamine_smiles2ecfp4_dict))

    smiles = (list(bb1_dict.keys()) + 
            list(bb2_dict.keys()) + 
            list(bb3_dict.keys()) + 
            list(enamine_smiles2ecfp4_dict.keys()))

    fps = (list(bb1_dict.values()) + 
        list(bb2_dict.values()) + 
        list(bb3_dict.values()) + 
        list(enamine_smiles2ecfp4_dict.values()))

    to_umap = pd.DataFrame({'SMILES': smiles, 'ECFP4s': fps, 'labels': mol_labels})
     
    # compute the embedding and plot        
    data = np.array(to_umap['ECFP4s'].tolist())

    # Now run UMAP
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(data)

    # Add the UMAP coordinates back to your dataframe
    to_umap['UMAP1'] = embedding[:, 0]
    to_umap['UMAP2'] = embedding[:, 1]

    output_out = ddr.cache.get_path(
        CacheNames.ANALOGS,
        filename=f"{output_prefix}_UMAP.parquet"
    )
    table = pa.Table.from_pandas(to_umap)
    pq.write_table(table, output_out)

    print(f'wrote UMAP to {output_out}')

    # plot the umap!
    color_map = {
        'bb1': 'magenta',
        'bb2': 'lightpink',
        'bb3': 'lightskyblue',
        'analog': 'lightgray'
    }

    umap_out = ddr.cache.get_path(
        CacheNames.ANALOGS,
        filename=f"{output_prefix}_UMAP.png"
    )
    
    fig, ax = plt.subplots()

    for label in to_umap['labels'].unique():
        subset = to_umap[to_umap['labels'] == label]
        ax.scatter(subset['UMAP1'], subset['UMAP2'], 
                c=color_map[label], label=label, alpha=0.8, s=20)

    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_title('UMAP Projection of ECFP4 Fingerprints')
    ax.legend()
    plt.savefig(umap_out)
    
    print(f'saved umap plot to {umap_out}')
    
    # compute tanimoto similarity
    matrixbb1, row_labelsbb1, col_labelsbb1 = compute_labeled_similarity(
        list(bb1_dict.values()), list(enamine_smiles2ecfp4_dict.values()),
        list(bb1_dict.keys()), list(enamine_smiles2ecfp4_dict.keys())
    )

    bb1_best_matches = get_best_matches(matrixbb1, row_labelsbb1, col_labelsbb1)
    
    matrixbb2, row_labelsbb2, col_labelsbb2 = compute_labeled_similarity(
        list(bb2_dict.values()), list(enamine_smiles2ecfp4_dict.values()),
        list(bb2_dict.keys()), list(enamine_smiles2ecfp4_dict.keys())
    )
    
    bb2_best_matches = get_best_matches(matrixbb2, row_labelsbb2, col_labelsbb2)
    
    matrixbb3, row_labelsbb3, col_labelsbb3 = compute_labeled_similarity(
        list(bb3_dict.values()), list(enamine_smiles2ecfp4_dict.values()),
        list(bb3_dict.keys()), list(enamine_smiles2ecfp4_dict.keys())
    )
    
    bb3_best_matches = get_best_matches(matrixbb3, row_labelsbb3, col_labelsbb3)
    
    # add the best matches smiles (for each bb1_ana, bb2_ana, bb3_ana) to the df_source dataframe, along with the tanimoto_1, tanimoto_2, tanimoto_3
    df_source = add_best_match_columns(
        df_source, 
        'buildingblock1_smiles', 
        bb1_best_matches, 
        'bb1_analog', 
        'analog1_tanimoto'
    )

    df_source = add_best_match_columns(
        df_source, 
        'buildingblock2_smiles', 
        bb2_best_matches, 
        'bb2_analog', 
        'analog2_tanimoto'
    )

    df_source = add_best_match_columns(
        df_source, 
        'buildingblock3_smiles', 
        bb3_best_matches, 
        'bb3_analog', 
        'analog3_tanimoto'
    )
    
    # then filter by any that have na values (need to create a map where na is written if smiles not in df)
    df_filtered = df_source.dropna()
    
    output_out = ddr.cache.get_path(
        CacheNames.ANALOGS,
        filename=f"{output_prefix}_similar_analogs.parquet"
    )
    table = pa.Table.from_pandas(df_filtered)
    pq.write_table(table, output_out)

    print(f'wrote predictions to {output_out}')

    

