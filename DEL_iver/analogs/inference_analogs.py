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

def inference_analog_moles(ddr, output_prefix):
    """
    Test the model trained in train_model() on the analog test set

    Parameters:
    -----------
    output_prefix : str
        **output_prefix should be the same output_prefix used for the gen_fingerprints() function and the train_model() function
        A string specifying the prefix for output files. 
        E.G if output_prefix='mymodel', the output model be named 'mymodel.png'
    
    Returns:
    --------
    parquet
        a file containing the output predictions on the analog set 
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

    input_in = ddr.cache.get_path(
        CacheNames.ANALOGS,
        filename=f"{output_prefix}_similar_analogs.parquet"
    )
    parquet_ans = pq.ParquetFile(input_in)
    testdf = parquet_ans.read().to_pandas()

    # get the bb fingerprint parquets
    bbs_to_fings = ddr.cache.get_path(
        CacheNames.ANALOGS,
        filename=f"{output_prefix}_enaminefingerprints.parquet"
    )

    # generate the building block dictionaries
    parquet_bb = pq.ParquetFile(bbs_to_fings)
    df_bb = parquet_bb.read().to_pandas()
    bb_dict = dict(zip(df_bb.iloc[:, 0], df_bb.iloc[:, 1]))
    del df_bb

    predictions_df = pd.DataFrame()

    target_test = testdf

    target_test_indices = target_test[['bb1_analog', 'bb2_analog', 'bb3_analog']].to_numpy()

    model_path = ddr.cache.get_path(
        CacheNames.MODELS,
        filename=f"{output_prefix}_trained_model.pth"
    )

    target_model = torch.load(model_path, weights_only=False)
    target_model.to(device)
    target_model.eval()

    batch_size = 10000
    total_samples = len(target_test_indices)
    print(f'total samples for inferencing: {(total_samples)}')
    all_predictions = []

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_indices = target_test_indices[start_idx:end_idx]
        
        batch_fps = []
        for index in batch_indices:
            fp1 = bb_dict[(index[0])]
            fp2 = bb_dict[(index[1])]
            fp3 = bb_dict[(index[2])]
            batch_fps.append(np.concatenate([fp1, fp2, fp3], axis=0))
        
        input_tensor = torch.tensor(np.array(batch_fps), dtype=torch.float32)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            outputs = target_model(input_tensor)
            predictions = torch.sigmoid(outputs)
        
        predictions_cpu = predictions.cpu().numpy()
        all_predictions.extend(predictions_cpu)

    target_test = target_test.copy()
    target_test['analog_pred_binding_probability'] = np.round(all_predictions, 3)

    # predictions_df = pd.concat([predictions_df, target_test], ignore_index=True)

    # print(f'number of predictions made {len(predictions_df)}')

    # Clean up GPU memory
    del target_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    output_out = ddr.cache.get_path(
        CacheNames.ANALOGS,
        filename=f"{output_prefix}_analog_predictions.parquet"
    )
    table = pa.Table.from_pandas(target_test)
    pq.write_table(table, output_out)

    print(f'wrote analog predictions to {output_out}')

