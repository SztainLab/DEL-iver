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
from pathlib import Path
from DEL_iver.utils.utils import *
from DEL_iver.utils.cache import CacheManager , CacheNames
from tqdm import tqdm



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


def split_data(ddr: "DEL.Data_Reader",output_dir:str = None,trainsplit:float = 0.80 ,validationsplit:float = 0.10 ,testsplit = 0.10 ,seed:int = 42):
    """
    Splits a DataFrame or file into train/validation/test and writes parquet files.
    """

    # Sanity check for split sizes
    if not 0 < trainsplit < 1:
        raise ValueError("`trainsplit` must be between 0 and 1.")
    if not 0 <= validationsplit < 1:
        raise ValueError("`validationsplit` must be between 0 and 1.")
    if trainsplit + validationsplit >= 1:
        raise ValueError("Sum of `trainsplit` and `validationsplit` must be less than 1.")

# Determine output directory
    prefix = ddr.source_file.stem
    t_pct = int(trainsplit * 100)
    v_pct = int(validationsplit * 100)

    filename = f"{prefix}.splits_t{t_pct}_v{v_pct}_seed{seed}.parquet"
    # --- Build single output path ---
    output_path = ddr.cache.get_path(CacheNames.SPLITS, filename=filename)
    
    pf = pq.ParquetFile(ddr.source_file)
    
    # Create a schema with a single integer column named "splits"
    schema = pa.schema([
        pa.field("splits", pa.int8())
    ])
    
    writer = pq.ParquetWriter(str(output_path), schema)

    rng = np.random.default_rng(seed)
    accumulated_tables = []
    try:

        for i in tqdm(range(pf.num_row_groups), desc="Generating Split Column"):

            num_rows = pf.metadata.row_group(i).num_rows
            
            # Generate random assignments (0=train, 1=val, 2=test)
            choices = rng.choice(
                [0, 1, 2],
                size=num_rows,
                p=[trainsplit, validationsplit, testsplit]
            )
            
            # Create a single-column PyArrow table for this chunk
            split_array = pa.array(choices, type=pa.int8())
            split_table = pa.Table.from_arrays([split_array], schema=schema)
            
            # Write the table
            writer.write_table(split_table)
            accumulated_tables.append(split_table)
            
    finally:
        # Essential: Close writer to finalize footers
        writer.close()
        
    splits_table = pa.concat_tables(accumulated_tables)

    return splits_table






def verify_data_split_feasibility(
    ddr, train_frac:float, val_frac:None, test_frac=None, min_rows_per_split=1
):

    """
    Checks that requested train/validation/test splits are valid and that each chunk
    in `actual_chunk_sizes` can accommodate the splits.

    Raises a ValueError if fractions are invalid or if any chunk is too small.
    """

    # Sanity checks for fractions


    if not 0 < train_frac < 1:
        raise ValueError("`train_frac` must be between 0 and 1.")
    if not 0 <= val_frac < 1:
        raise ValueError("`val_frac` must be between 0 and 1.")
    if train_frac + val_frac >= 1:
        raise ValueError("Sum of `train_frac` and `val_frac` must be less than 1.")
    if test_frac <= 0:
        raise ValueError("`test_frac` must be positive after accounting for train and val fractions.")


    bad_chunks = []

    for i, n_rows in enumerate(ddr.chunk_size):
        n_train = int(n_rows * train_frac)
        n_val = int(n_rows * val_frac)
        n_test = n_rows - n_train - n_val  # ensures total rows accounted for

        if any(x < min_rows_per_split for x in [n_train, n_val, n_test]):
            bad_chunks.append(
                f"Chunk {i} ({n_rows} rows) -> train={n_train}, val={n_val}, test={n_test}"
            )

    if bad_chunks:
        msg = "The following chunks are too small for the requested splits:\n" + "\n".join(bad_chunks)
        raise ValueError(msg)

    return True  # All checks passed

#!TODO: AS IT IS NOW, THE ENTIRE DATA GETS LOADED INTO MEMORY , NEED TO FIX BUT IT WORKS WITH SMALL TEST DAT SET
def split_data1(ddr: "DEL.Data_Reader", prefix:str = None,output_dir:str = None,trainsplit:float = 0.80 ,validationsplit:float = 0.10,testsplit:float=0.10): #TODO: BUILD IN A CHECK TO SEE IF SPLITTING DATA RESULTS IN NOTHING, FIGURE OUT WHY IT CRASHED WITH CHUNK SIZE 5, I ASSUME IT COULD NOT DO THE 3 WAY SPLIT WITH HOW MUCH DATA WAS PER ROW
    """

    """


    status=verify_data_split_feasibility(ddr,train_frac=trainsplit,val_frac=validationsplit,test_frac=testsplit)

    data=ddr.data




    if output_dir:
                train_df.to_parquet(f"{output_dir}/{prefix}_train.parquet")
                val_df.to_parquet(f"{output_dir}/{prefix}_val.parquet")
                test_df.to_parquet(f"{output_dir}/{prefix}_test.parquet")
                
    return train_df, val_df, test_df



            

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
