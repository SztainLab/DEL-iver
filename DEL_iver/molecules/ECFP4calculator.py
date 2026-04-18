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

def gen_fingerprints(ddr, output_prefix, chunk_size=500000, ecfp4_size=1024, remove_dy=False):
    """
    Calculate ECFP4 fingerprints from a parquet file containing SMILES strings.
    
    Parameters:
    -----------
    output_prefix : str
        A string specifying the prefix for output files. 
        E.G if output_prefix='mybbs', the output would be named 'mybbs_bb1_fingerprints.parquet'
    chunk_size : int, default=500000
        Chunk size for reading the input parquet file
    ecfp4_size : int, default=1024
        Size of ECFP4 fingerprints
    remove_dy : bool, default=False
        Whether to remove Dy tag and replace with PEG linker
    
    Returns:
    --------
    parquet file 
        parquet files containing IDs and ECFP4 fingerprints, one parquet for each bb set
    
    Example:
    --------
    >>> gen_fingerprints(
            ddr,
    ...     output_prefix="experiment1",
    ...     chunk_size=1000000,
    ...     ecfp4_size=2048
    ... )
    """
    
    print(f'Using {multiprocessing.cpu_count()} CPUs...')
    print(f'Output prefix: {output_prefix}')
    print(f'Chunk size: {chunk_size}')
    print(f'ECFP4 size: {ecfp4_size}')
    print(f'Remove Dy: {remove_dy}')
    
    filename = ddr.cache.get_path(
        CacheNames.BB_DICTIONARIES,
        filename=f"{CacheNames.BB_DICTIONARIES.value}.{ddr.source_file.stem}_id_to_smiles.parquet"
    )

    filename_map = ddr.cache.get_path(
        CacheNames.BB_DICTIONARIES,
        filename=f"{CacheNames.BB_DICTIONARIES.value}.{ddr.source_file.stem}.parquet"
    )

    # print(filename)
    # Read parquet file in batches
    parquet_file = pq.ParquetFile(filename)
    all_fps = []
    all_ids = []
    for batch in tqdm(parquet_file.iter_batches(batch_size=chunk_size), desc="Processing batches"):
        chunk = batch.to_pandas()
        print(chunk.head())
        cs = list(chunk.columns)
        
        # Apply Dy replacement if requested
        if remove_dy:
            chunk[cs[1]] = chunk[cs[1]].apply(replace_Dy)
        
        # Compute fingerprints for each SMILES string
        print(cs[1][0])

        fps = [retrieve_mol_fp(smi, 'ECFP4', ecfp4_size) for smi in chunk[cs[1]]]
        ids = chunk[cs[0]].tolist()
        
        all_fps.extend(fps)
        all_ids.extend(ids)
    
    # Create DataFrame with IDs and fingerprints
    result_df = pd.DataFrame({
        'id': all_ids,
        'ecfp4_fingerprint': all_fps
    })

    parquet_map = pq.ParquetFile(filename_map)
    df_map = parquet_map.read().to_pandas()
    # print(df_map.columns)

    bb_cols = list(ddr.building_blocks)

    bb1_pos_ids = list(df_map[f'{bb_cols[0]}_positional_id'])
    bb2_pos_ids = list(df_map[f'{bb_cols[1]}_positional_id'])
    bb3_pos_ids = list(df_map[f'{bb_cols[2]}_positional_id'])
    
    bb1_df = result_df[result_df['id'].isin(bb1_pos_ids)]
    bb2_df = result_df[result_df['id'].isin(bb2_pos_ids)]
    bb3_df = result_df[result_df['id'].isin(bb3_pos_ids)]

    for bname, df in zip(['bb1', 'bb2', 'bb3'], [bb1_df, bb2_df, bb3_df]):
        # Save to parquet file
        output_out = ddr.cache.get_path(
            CacheNames.SMILESEMBEDDING,
            filename=f"{output_prefix}_{bname}_fingerprints.parquet"
        )
    
        # Convert to PyArrow Table and save as Parquet
        table = pa.Table.from_pandas(df)
        print(len(df))
        pq.write_table(table, (output_out))
        print(f"Fingerprints saved to: {output_out}")

    print(f"Total fingerprints generated: {len(result_df)}")

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