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

import sys

# defining the custom dataloader class
class TrainBBFPDataset_v1(Dataset):
    def __init__(self, indices, bbs_1_fps, bbs_2_fps, bbs_3_fps, labels):
        self.indices = indices
        self.bbs_1_fps = bbs_1_fps
        self.bbs_2_fps = bbs_2_fps
        self.bbs_3_fps = bbs_3_fps
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        index = self.indices[idx]
        
        # Fetch the fingerprints for each building block type
        fp1 = torch.tensor(self.bbs_1_fps[int(index[0])], dtype=torch.float32)
        fp2 = torch.tensor(self.bbs_2_fps[int(index[1])], dtype=torch.float32)
        fp3 = torch.tensor(self.bbs_3_fps[int(index[2])], dtype=torch.float32)

        # Concatenate the fingerprints
        concatenated_fp = torch.cat((fp1, fp2, fp3), dim=0)
        
        return concatenated_fp, torch.tensor(self.labels[idx], dtype=torch.float32)

class AttentionModule(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModule, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, inputs):
        weights = F.softmax(self.attention_weights(inputs), dim=0)
        return (inputs * weights).sum(dim=0)

class BBFP_PermInvarNN_v3(nn.Module): # Neural network for permutation invariant
    def __init__(self, fingerprint_length):
        super(BBFP_PermInvarNN_v3, self).__init__()
        self.fp_length = fingerprint_length

        # Shared layers for the second and third building blocks
        self.shared_fc = nn.Linear(fingerprint_length, fingerprint_length)
        # Individual layer for the first building block
        self.fc1_bb1 = nn.Linear(fingerprint_length, fingerprint_length)
        # Attention layer
        self.attention = AttentionModule(fingerprint_length)
        
        # Following layers process the combined information
        self.fc2 = nn.Linear(fingerprint_length * 2, fingerprint_length)  # Input size doubled due to combination of bb1 and aggregated bb2&3
        self.fc3 = nn.Linear(fingerprint_length, fingerprint_length // 2)
        self.fc4 = nn.Linear(fingerprint_length // 2, fingerprint_length // 4)
        self.fc5 = nn.Linear(fingerprint_length // 4, 1)

    def forward(self, x):
        # Split the concatenated input tensor into three parts
        bb1 = x[:, :self.fp_length]  # First bits for bb1
        bb2 = x[:, self.fp_length:self.fp_length*2]  # Next bits for bb2
        bb3 = x[:, self.fp_length*2:]  # Last bits for bb3

        # Process bb1 through its dedicated layer
        bb1_processed = F.relu(self.fc1_bb1(bb1))

        # Apply shared layer to bb2 and bb3
        bb2_processed = F.relu(self.shared_fc(bb2))
        bb3_processed = F.relu(self.shared_fc(bb3))

        # Use attention on bb2 and bb3
        aggregated_bb2_bb3 = self.attention(torch.stack([bb2_processed, bb3_processed]))

        # Concatenate bb1_processed and aggregated_bb2_bb3
        combined_features = torch.cat((bb1_processed, aggregated_bb2_bb3), dim=1)

        # Further processing the combined features
        x = F.relu(self.fc2(combined_features))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

def train_model(ddr, output_prefix):
    """
    Train a model on building block ECFP4 fingerprints to predict binding probability of molecules
    Before training, the dataset is split into train and test (80/20) splits

    Parameters:
    -----------
    output_prefix : str
        **output_prefix should be the same output_prefix used for the gen_fingerprints() function
        A string specifying the prefix for output files. 
        E.G if output_prefix='mymodel', the output model be named 'mymodel.pth'
    
    Returns:
    --------
    model
        a model (.pth) file trained on the training set
    parquet file
        parquet files specifying the train and test splits used 
    
    Example:
    --------
    >>> train_model(
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

    # get the original dataframe 
    source_file = ddr.source_file
    parquet_source = pq.ParquetFile(source_file)
    df_source = parquet_source.read().to_pandas()

    bindlabels = list(df_source[ddr.label])
    del df_source

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

    # read in the bb positional df as well
    filename = ddr.cache.get_path(
        CacheNames.BB_DICTIONARIES,
        filename=f"{CacheNames.BB_DICTIONARIES.value}.{ddr.source_file.stem}.parquet"
    )
    parquet_alldata = pq.ParquetFile(filename)
    all_data_df = parquet_alldata.read().to_pandas()

    # print(all_data_df.columns)
    # print(all_data_df.head())
    # print(len(all_data_df))
    # print(len(bindlabels))

    all_data_df = all_data_df[['buildingblock1_smiles_positional_id', 'buildingblock2_smiles_positional_id', 'buildingblock3_smiles_positional_id']]

    # print(all_data_df.columns)
    # generate train and test splits, then write to parquet files
    trainx, testx, trainlabel, testlabel = train_test_split(all_data_df, bindlabels, train_size=0.2, random_state=42)

    trainx['binds'] = list(trainlabel)
    testx['binds'] = list(testlabel)

    trainx = trainx[trainx['buildingblock1_smiles_positional_id'].isin(list(bb1_dict.keys()))]
    trainx = trainx[trainx['buildingblock2_smiles_positional_id'].isin(list(bb2_dict.keys()))]
    trainx = trainx[trainx['buildingblock3_smiles_positional_id'].isin(list(bb3_dict.keys()))]

    testx = testx[testx['buildingblock1_smiles_positional_id'].isin(list(bb1_dict.keys()))]
    testx = testx[testx['buildingblock2_smiles_positional_id'].isin(list(bb2_dict.keys()))]
    testx = testx[testx['buildingblock3_smiles_positional_id'].isin(list(bb3_dict.keys()))]

    output_out = ddr.cache.get_path(
        CacheNames.MODELS,
        filename=f"{output_prefix}_trainset.parquet"
    )
    table = pa.Table.from_pandas(trainx)
    pq.write_table(table, output_out)

    output_out2 = ddr.cache.get_path(
        CacheNames.MODELS,
        filename=f"{output_prefix}_testset.parquet"
    )
    table = pa.Table.from_pandas(testx)
    pq.write_table(table, output_out2)

    print(f'wrote train and test sets to: {output_out} and {output_out2}')

    all_bb_indices = trainx[['buildingblock1_smiles_positional_id', 'buildingblock2_smiles_positional_id', 'buildingblock3_smiles_positional_id']].to_numpy()

    batch_size = 10000
    target_labels = trainx[ddr.label].to_numpy()

    target_dataset = TrainBBFPDataset_v1(all_bb_indices, bb1_dict, bb2_dict, bb3_dict, target_labels)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

    model = BBFP_PermInvarNN_v3(1024).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        '''
        # Randomly select negative samples, you can adjust the size ratio as needed
        sampled_negative = np.random.choice(len(negative_indices), size=len(positive_indices), replace=False)
        batch_indices = np.concatenate((positive_indices, [negative_indices[x] for x in sampled_negative]))
        batch_labels = np.zeros(len(batch_indices), dtype=int)
        batch_labels[:len(positive_indices)] = 1

        # Create dataset and dataloader with the selected indices
        target_dataset = TrainBBFPDataset_v1(batch_indices, bbs_1_fp, bbs_2_fp, bbs_3_fp, batch_labels)
        target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
        '''

        running_loss = 0.0
        for i, (batch_fp, batch_labels) in enumerate(target_dataloader):
            batch_fp = batch_fp.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            outputs = model(batch_fp).squeeze()
            loss = criterion(outputs, batch_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print average loss every 100 batches
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # Print loss after each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}] completed, Loss: {running_loss / len(target_dataloader)}')

    output_out = ddr.cache.get_path(
        CacheNames.MODELS,
        filename=f"{output_prefix}_trained_model.pth"
    )
    torch.save(model, output_out)
    print(f'wrote model to {output_out}')
    

if __name__ == '__main__':
    print("""
    This script is designed to be imported and called as a function.
    
    Example usage:
    
    from this_script_name import gen_fingerprints
    
    df = gen_fingerprints(
        filename="path/to/your/data.parquet",
        output_prefix="my_experiment",
        chunk_size=500000,
        ecfp4_size=1024,
        remove_dy=False
    )
    """)


