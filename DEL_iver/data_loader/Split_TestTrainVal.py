#!/bin/python
import os 
import numpy as np
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#import pickle
from sklearn.model_selection import train_test_split
import argparse
#import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
#from joblib import Parallel, delayed
#import multiprocessing
#import tqdm

from DEL_iver.utils.utils import *



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

#!currently chunk size is not used in this version of the function but i have not removed it because i dont want to make too many chnanges at once
def split_data(df:pd.DataFrame,prefix:str = None,output_dir:str = None,trainsplit:float = 0.80 ,validationsplit:float = 0.10 ,chunk_dize:int = 500000 ): #TODO: BUILD IN A CHECK TO SEE IF SPLITTING DATA RESULTS IN NOTHING, FIGURE OUT WHY IT CRASHED WITH CHUNK SIZE 5, I ASSUME IT COULD NOT DO THE 3 WAY SPLIT WITH HOW MUCH DATA WAS PER ROW
    """
    Splits a DataFrame or file into train/validation/test and writes parquet files.
    """

    if df is None:
        raise ValueError("Must provide df from Data_Reader")
    # Sanity check for split sizes
    if not 0 < trainsplit < 1:
        raise ValueError("`trainsplit` must be between 0 and 1.")
    if not 0 <= validationsplit < 1:
        raise ValueError("`validationsplit` must be between 0 and 1.")
    if trainsplit + validationsplit >= 1:
        raise ValueError("Sum of `trainsplit` and `validationsplit` must be less than 1.")

    #Handles getting an entire data frame 
    if isinstance(df, pd.DataFrame):
        # split the dataframe into train, validation, and test sets
        train_df, temp_df = train_test_split(df, train_size=trainsplit, random_state=42)
        val_df, test_df = train_test_split(temp_df, train_size=validationsplit, random_state=42) 
        
        # write the train, validation, test splits to parquet files
        if output_dir: 
            os.makedirs(output_dir,exist_ok=True)
            if prefix is None:
                prefix = "data"
            train_df.to_parquet(f'{output_dir}/{prefix}_trainset.parquet')
            val_df.to_parquet(f'{output_dir}/{prefix}_validationset.parquet')
            test_df.to_parquet(f'{output_dir}/{prefix}_testset.parquet')

        return train_df, val_df, test_df
            
        
    #Handles needing to iterate over chunks
    elif hasattr(df, "__iter__") and not isinstance(df, pd.DataFrame):
        train_list = []
        val_list = []
        test_list = []
        for chunk in df:
            train_df, temp_df = train_test_split(chunk, train_size=trainsplit, random_state=42)
            val_df, test_df = train_test_split(temp_df,train_size=validationsplit,random_state=42,)
            train_list.append(train_df)
            val_list.append(val_df)
            test_list.append(test_df)


        train_df = pd.concat(train_list, ignore_index=True)
        val_df = pd.concat(val_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)

        if output_dir:
                    train_df.to_parquet(f"{output_dir}/{prefix}_train.parquet")
                    val_df.to_parquet(f"{output_dir}/{prefix}_val.parquet")
                    test_df.to_parquet(f"{output_dir}/{prefix}_test.parquet")
                    
        return train_df, val_df, test_df
    

    else:
        raise ValueError("df must be either a pandas DataFrame or a pandas TextFileReader (iterator from csv)")

            

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

    #i moved the code from main into its own function, and i call it
    split_data(df=df,prefix=args.prefix,output_dir=args.output_dir,trainsplit=args.trainsplit,validationsplit=args.validationsplit,chunk_size=args.chunk_size)

    
if __name__ == '__main__':
    main()
