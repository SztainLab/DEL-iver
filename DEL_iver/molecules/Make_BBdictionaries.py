#!/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import argparse
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pyarrow.compute as pc
from joblib import Parallel, delayed
import multiprocessing
import tqdm
from tqdm import tqdm
from itertools import combinations
import itertools
from DEL_iver.utils.utils import *
from DEL_iver.utils.cache import CacheManager, CacheNames

class HelpAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        help_text = """
Purpose: Create dictionaries of building block and fullmolecule fingerprints and store as pickle files. 
Also generate a new csv file for this dataset, with molecule smiles and fingerprints removed. 
Storing the fingerprints in dictionaries and identifying molecules and building blocks by IDs rather smiles saves space and time during training. 

Usage: python Make_BBdictionaries.py <filename> <output_dir> <prefix> [options]

Arguments:
    filename    - the parquet file output from running ECFP4_calculator.py
    output_dir  - output directory path
    prefix      - prefix for output pickle and parquet files (e.g. ecfp4_1024_)

Options:
    -c, --chunk_size INT                                          - Chunk size for parquet reading (default: 500000)
    -b, --bb_fingerprints                                         - If specified, generate building block fingerprint dictionaries 
    -l, --bb_list (space delimited list of bb ecfp4 column names) - Must be specified if option -b is specified 
    -h, --help                                                    - Show this help message
"""
        print(help_text)
        sys.exit(0)



def _make_bb_smiles_to_id_dict(source_file,building_blocks):
    pf = pq.ParquetFile(source_file)
    pf= pf.read(columns=building_blocks)

    all_smiles = set()
    for block in tqdm(building_blocks,desc="Finding unique building blocks"):
        all_smiles.update(pc.unique(pf[block]).to_pylist())
    smile_to_id = {smile: idx for idx, smile in enumerate(all_smiles)}
    return smile_to_id 
    

def _assign_id_per_row(source_file,building_blocks,smile_to_id):
    pf = pq.ParquetFile(source_file)
    pf= pf.read(columns=building_blocks)
    col_arrays = {}
    for block in tqdm(building_blocks, desc="Assigning to table"):
        encoded = pc.dictionary_encode(pf[block].cast(pa.large_string()).combine_chunks())
        block_dict = encoded.dictionary.to_pylist()
        # Map block-local dictionary to global IDs (small, only unique values)
        local_to_global = pa.array([smile_to_id[s] for s in block_dict], type=pa.int32())
        # Vectorized index remapping — no Python loop over rows
        col_arrays[f"{block}_id"] = pc.take(local_to_global, encoded.indices)

    table = pa.table(col_arrays)

    return table



#TODO: USE GROUPBY LOGIC 
def _assign_disynthon_ids(table, building_blocks):
    bb_id_cols = [f"{bb}_id" for bb in building_blocks]
    col_arrays = {}
    combos = list(combinations(range(len(building_blocks)), 2))

    for combo in tqdm(combos, desc="Assigning disynthon ids"):
        col_names = [bb_id_cols[i] for i in combo]
        combo_label = "_".join(str(i + 1) for i in combo)

        # Combine columns into a single string key e.g. "12_345"
        key = pc.binary_join_element_wise(
            *[pc.cast(table[col].combine_chunks(), pa.string()) for col in col_names],
            "_"
        )
        encoded = pc.dictionary_encode(key)
        col_arrays[f"disynthon_{combo_label}_id"] = encoded.indices.cast(pa.int32())

    return pa.table({**{c: table[c] for c in table.schema.names}, **col_arrays})


def generate_bb_dictionaries(ddr):


    source_file=ddr.source_file
    building_blocks=ddr.building_blocks

    output_path = ddr.cache.get_path(
        CacheNames.BB_DICTIONARIES,
        filename=f"{CacheNames.BB_DICTIONARIES.value}.{ddr.source_file.stem}.parquet"
    )


    #TODO: check if building_blocks list is in the column od source_file if not raise error, this should be handle by cache_manager when instantiating the data reader

    #TODO: if output already exists for this cache, then skip this step

    source_file=ddr.source_file
    building_blocks=ddr.building_blocks


    smile_to_id=_make_bb_smiles_to_id_dict(source_file,building_blocks)
    id_to_smile = {idx: smile for smile, idx in smile_to_id.items()}


    id_to_smile_table = pa.table({
        "id": list(id_to_smile.keys()),
        "smiles": list(id_to_smile.values())
    })

    pq.write_table(
        id_to_smile_table,
        output_path.with_name(output_path.stem + "_id_to_smiles.parquet")
    )

    table=_assign_id_per_row(source_file,building_blocks,smile_to_id)

    table=_assign_disynthon_ids(table,building_blocks)

    pq.write_table(table, output_path) #TODO: writting should be handled by the cache manager

    #print(table)



    return table, id_to_smile

## now just compute pbind and enrichment









    









def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="the parquet file output from running ECFP4_calculator.py")
    parser.add_argument('output_dir', help="path to the output directory")
    parser.add_argument('prefix', help='prefix for output pickle and parquet files (e.g. ecfp4_1024)')
    parser.add_argument('-c', '--chunk_size', type=int, default=500000, help='change the chunk_size for reading in the input csv (default is 500,000)')
    parser.add_argument('b', '--bb_fingerprints', action='store_true', help='if specified, building block dictionaries will be generated in addition to the full molecule fingerprints')
    parser.add_argument('-l', '--bb_list', nargs='+', default=None, help='space delimited list of bb ecfp4 column names) - Must be specified if option -b is specified')
    parser.add_argument('-h', '--help', action=HelpAction)
    
    args = parser.parse_args()
    
    print(f'Using {multiprocessing.cpu_count()} CPUs...')

    # Read parquet in chunks and write to Parquet in batches
    if args.chunk_size != 500000:
        chunk_size = args.chunk_size
    else:
        chunk_size = 500000
        
    reader = pd.read_parquet(args.filename, chunksize=chunk_size)
    
    full_molecule_dict = {}
    jfull = 0
    bb_dicts = {}
    jbbs = {}
    
    if args.bb_fingerprints == True:
        if args.bb_list == None:
            parser.error("If -b is specified, a list of building block ecfp4 column names must be provided")
        else:
            n_bbs = len(args.bb_list)
            # make the bb fp dictionaries here as well
            for colname in list(args.bb_list):
                bb_dicts[colname] = {}
                jbbs[colname] = 0
    
    for i, chunk in tqdm(enumerate(reader)):
        print(f'Processing chunk {i}')
        bbsmilesdicts = {}
        
        # generate full molecule smiles to integer id dictionary
        smiles = set(list(chunk['molecule_smiles']))
        smile_dict = {}
        for smile in smiles:
            smile_dict[smile] = jfull
            jfull += 1
        
        # if true, continue with the same process for building blocks
        if args.bb_fingerprints == True:
            # process the building block dictionaries too
            for k in range(len(list(args.bb_list))):
                colname = list(args.bb_list)[k]
                bbnum = k+1
                smiles = set(list(chunk[f'building_block{bbnum}_smiles']))
                bb_smiles_dict = {}
                for smile in smiles:
                    bb_smiles_dict[smile] = jbbs[colname]
                    jbbs[colname] += 1
                
                bbsmilesdicts[colname] = bb_smiles_dict
                
        # generate id list from molecule smiles to integer id dictionary
        df = chunk.copy()
        df['molecule_id'] = df['smiles'].map(smile_dict) 
        chunk_fullmole_ecfp4_dict = dict(zip(df['molecule_id'], df['fullmolecule_ecfp4_fp']))
        full_molecule_dict |= chunk_fullmole_ecfp4_dict
        
        # remove the ecfp4 fp column
        df = df.drop('fullmolecule_ecfp4_fp', axis=1)
        # remove the molecule smiles column
        df = df.drop('molecule_smiles', axis=1)
        
        if args.bb_fingerprints == True:
            # write the building block id columns as well
            for colname in list(bbsmilesdicts.keys()):
                newname = f'{colname[:-9]}_id'
                smilesname = f'{colname[:-9]}_smiles' 
                df[newname] = df[smilesname].map(bbsmilesdicts[colname])
                
                bbdict_idecfp4 = dict(zip(df[newname], df[colname]))
                bb_dicts[colname] |= bbdict_idecfp4
            
            # remove the remaining smiles and ecfp4 columns
            df = df.drop(colname, axis=1)
            df = df.drop(smilesname, axis=1)
            
            
    # make output directory if it doesn't already exist
    os.makedirs(args.output_dir, exist_ok=True) 
    
    # write the bb --> ecfp4 dictionaries to pickle files    
    with open(f'{args.output_dir}/{args.prefix}_fullmolecule_dict.pkl', 'wb') as f:
        pickle.dump(full_molecule_dict, f)
        
    for colname, coldict in bb_dicts.items():
        with open(f'{args.output_dir}/{args.prefix}_{colname}.pkl', 'wb') as f:
            pickle.dump(coldict, f)
    
    # also write a new parquet file that has all smiles and ecfp4 columns removed
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, root_path=args.output_dir, compression='snappy')

if __name__ == '__main__':
    main()


