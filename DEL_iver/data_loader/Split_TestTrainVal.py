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
from DEL_iver.utils.cache import get_cache_path, is_cached, clear_cache
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
    if output_dir is None:
        # Assuming get_cache_path logic
        cache_src = Path(ddr.source_file).parent / "splits"
    else:
        cache_src = Path(output_dir)

    # Define file paths

    prefix = ddr.source_file.stem
    paths = {
        "train": cache_src / f"{prefix}.train.parquet",
        "val": cache_src / f"{prefix}.validation.parquet",
        "test": cache_src / f"{prefix}.test.parquet"
    }

    cache_src.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(ddr.source_file)
    schema = pf.schema.to_arrow_schema()

    writers = {
            "train": pq.ParquetWriter(str(paths["train"]), schema),
            "val": pq.ParquetWriter(str(paths["val"]), schema),
            "test": pq.ParquetWriter(str(paths["test"]), schema)
        }

    try:
        # 6. Process by Row Group (Faster than small chunks)
        for i in tqdm(range(pf.num_row_groups), desc="Splitting Row Groups"):
            table = pf.read_row_group(i)
            num_rows = table.num_rows
            
            # Generate random assignments (0=train, 1=val, 2=test)
            # Using a seed ensures reproducibility if needed
            rng = np.random.default_rng(seed)
            choices = rng.choice(
                [0, 1, 2],
                size=num_rows,
                p=[trainsplit, validationsplit, testsplit]
            )

            # Filter and Write only if the split contains rows
            for idx, key in enumerate(["train", "val", "test"]):
                split_table = table.filter(pa.array(choices == idx))
                if split_table.num_rows > 0:
                    writers[key].write_table(split_table)

    finally:
        # Essential: Close all writers to finalize footers
        for w in writers.values():
            w.close()

                


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
