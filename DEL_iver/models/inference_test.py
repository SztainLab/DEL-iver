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
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import random
from itertools import combinations
import itertools
from DEL_iver.utils.utils import *
from DEL_iver.utils.cache import CacheManager, CacheNames
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import torch
from torch.utils.data import DataLoader
import random
from sklearn.model_selection import train_test_split
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
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve

def inference(ddr, output_prefix):
    """
    Test the model trained in train_model() on the test set also generated in train_model().

    Parameters:
    -----------
    output_prefix : str
        **output_prefix should be the same output_prefix used for the gen_fingerprints() function and the train_model() function
        A string specifying the prefix for output files. 
        E.G if output_prefix='mymodel', the output model be named 'mymodel.png'
    
    Returns:
    --------
    parquet
        a file containing the output predictions on the test set
    png
        png of AUROC plot (performance of model on test set)
        png of Precision recall plot (performance of model on test set)    
    
    Example:
    --------
    >>> inference(
            ddr,
    ...     output_prefix="experiment1",
    ... )
        """

    # Check if CUDA (GPU support) is available and set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: cuda, with {torch.cuda.device_count()} GPUs available")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    fingerprint_length = 1024

    print(f'Output prefix: {output_prefix}')

    # load data to test on
    testf = ddr.cache.get_path(
        CacheNames.MODELS,
        filename=f"{output_prefix}_testset.parquet"
    )

    testparq = pq.ParquetFile(testf)
    testdf = testparq.read().to_pandas()

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

    # generate the building block dictionaries
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

    predictions_df = pd.DataFrame()

    target_test = testdf

    target_test_indices = target_test[['buildingblock1_smiles_positional_id', 'buildingblock2_smiles_positional_id', 'buildingblock3_smiles_positional_id']].to_numpy()

    model_path = ddr.cache.get_path(
        CacheNames.MODELS,
        filename=f"{output_prefix}_trained_model.pth"
    )

    target_model = torch.load(model_path, weights_only=False)
    target_model.to(device)
    target_model.eval()

    batch_size = 10000
    total_samples = len(target_test_indices)
    print(f'total samples for testing: {(total_samples)}')
    all_predictions = []

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_indices = target_test_indices[start_idx:end_idx]
        
        batch_fps = []
        for index in batch_indices:
            fp1 = bb1_dict[int(index[0])]
            fp2 = bb2_dict[int(index[1])]
            fp3 = bb3_dict[int(index[2])]
            batch_fps.append(np.concatenate([fp1, fp2, fp3], axis=0))
        
        input_tensor = torch.tensor(np.array(batch_fps), dtype=torch.float32)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            outputs = target_model(input_tensor)
            predictions = torch.sigmoid(outputs)
        
        predictions_cpu = predictions.cpu().numpy()
        all_predictions.extend(predictions_cpu)

    target_test = target_test.copy()
    target_test['pred'] = np.round(all_predictions, 3)

    predictions_df = pd.concat([predictions_df, target_test], ignore_index=True)

    print(f'number of predictions made {len(predictions_df)}')

    # Clean up GPU memory
    del target_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    output_out = ddr.cache.get_path(
        CacheNames.MODELS,
        filename=f"{output_prefix}_testset_predictions.parquet"
    )
    table = pa.Table.from_pandas(predictions_df)
    pq.write_table(table, output_out)

    print(f'wrote predictions to {output_out}')

    y_true = predictions_df['binds']
    y_scores = predictions_df['pred']

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    recall, precision, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    ap_baseline = sum(y_true)/len(y_true)

    auroc_out = ddr.cache.get_path(
        CacheNames.MODELS,
        filename=f"{output_prefix}_AUROC_plot.png"
    )

    pr_out = ddr.cache.get_path(
        CacheNames.MODELS,
        filename=f"{output_prefix}_PR_plot.png"
    )

    plt.figure()
    plt.plot([0, 1], [0, 1], color='gray', lw=5, linestyle='--', label=f'random AUROC 0.5')
    plt.plot(fpr, tpr, color='magenta', lw=5, label=f'AUROC {roc_auc:.5f})')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{output_prefix} Test Performance (AUROC)')
    plt.legend(loc="lower right", fancybox=True, framealpha=1.0)
    plt.savefig(auroc_out)
    print(f'auroc plot saved to {auroc_out}')

    plt.cla()

    plt.figure()
    plt.axhline(y=ap_baseline, color='gray', lw=5, linestyle='--', label=f'baseline AP {ap_baseline}')
    plt.plot(recall, precision, color='lightskyblue', lw=5, label=f'AP {ap:.5f})')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{output_prefix} Test Performance (PR)')
    plt.legend(loc="lower right", fancybox=True, framealpha=1.0)
    plt.savefig(pr_out)
    print(f'precision recall saved to {pr_out}')



