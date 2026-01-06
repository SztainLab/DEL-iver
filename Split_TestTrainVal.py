#!/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.model_selection import train_test_split
import argparse
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import tqdm

from utils import *

class HelpAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        help_text = """
Purpose: Split the parquet file output from Make_BBdictionaries.py into train, test, and validation sets,
and output each to a parquet file

Usage: python Split_TestTrainVal.py <filename> <output_dir> <prefix> [options]

Arguments:
    filename    - the parquet file output from running Make_BBdictionaries.py
    output_dir  - output directory path
    prefix      - prefix for output parquet files

Options:
    -c, --chunk_size INT        - Chunk size for parquet reading (default: 500000)
    -t, --trainsplit FLOAT      - fraction of data for train (default: 0.80). The fraction of test will be 1.0 - trainsplit - validationsplit.
    -v, --validationsplit FLOAT - fraction of data for validation (default: 0.10). The fraction of test will be 1.0 - trainsplit - validation split. 
    -h, --help                  - Show this help message
"""
        print(help_text)
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="the parquet file output from running Make_BBdictionaries.py")
    parser.add_argument('output_dir', help="path to the output directory")
    parser.add_argument('prefix', help='prefix for output parquet files')
    parser.add_argument('-c', '--chunk_size', type=int, default=500000, help='change the chunk_size for reading in the input csv (default is 500,000)')
    parser.add_argument('-t', '--trainsplit', type=float, default=0.80, help='fraction of data for train (default: 0.80). The fraction of test will be 1.0 - trainsplit - validationsplit.')
    parser.add_argument('v', '--validationsplit', type=float, default=0.10, help='fraction of data for validation (default: 0.10). The fraction of test will be 1.0 - trainsplit - validationsplit.')
    parser.add_argument('-h', '--help', action=HelpAction)
    
    args = parser.parse_args()    

    df = pd.read_parquet(args.filename)
    
    # split the dataframe into train, validation, and test sets
    train_df, temp_df = train_test_split(df, train_size=args.trainsplit, random_state=42)
    val_df, test_df = train_test_split(temp_df, train_size=args.validationsplit, random_state=42) 
    
    # write the train, validation, test splits to parquet files
    train_df.to_parquet(f'{args.output_dir}/{args.prefix}_trainset.parquet')
    val_df.to_parquet(f'{args.output_dir}/{args.prefix}_validationset.parquet')
    test_df.to_parquet(f'{args.output_sir}/{args.prefix}_testset.parquet')
    
if __name__ == '__main__':
    main()
