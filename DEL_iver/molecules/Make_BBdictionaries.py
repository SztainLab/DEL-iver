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
from joblib import Parallel, delayed
import multiprocessing
import tqdm
from tqdm import tqdm

from DEL_iver.utils.utils import *

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

#todo include disinthons
def generate_BB_dictionaries(ddr, bb_fingerprints: bool):
    
    # full molecule maps
    full_molecule_smile_to_int = {}
    full_molecule_int_to_smile = {}
    jfull = 0

    # bb maps: one forward + one reverse dict per bb column
    bb_smile_to_int = {}
    bb_int_to_smile = {}
    jbbs = {}

    bb_list = ddr.building_blocks      # e.g. ['col_2', 'col_3', 'col_5']
    reader = ddr.data
    molecule_smiles = ddr.molecule_smiles  # e.g. 'col_6'

    if bb_fingerprints:
        for colname in bb_list:
            bb_smile_to_int[colname] = {}
            bb_int_to_smile[colname] = {}
            jbbs[colname] = 0

    for i, chunk in tqdm(enumerate(reader)):

        # --- full molecule ---
        for smile in chunk[molecule_smiles].unique():
            if smile not in full_molecule_smile_to_int:
                full_molecule_smile_to_int[smile] = jfull
                full_molecule_int_to_smile[jfull] = smile
                jfull += 1

        # --- building blocks ---
        if bb_fingerprints:
            for colname in bb_list:
                for smile in chunk[colname].unique():
                    if smile not in bb_smile_to_int[colname]:
                        idx = jbbs[colname]
                        bb_smile_to_int[colname][smile] = idx
                        bb_int_to_smile[colname][idx] = smile
                        jbbs[colname] += 1


    #TODO: maybe make it return a data frame or something simple
    if bb_fingerprints:
        return (full_molecule_smile_to_int, full_molecule_int_to_smile,
                bb_smile_to_int, bb_int_to_smile)
    else:
        return full_molecule_smile_to_int, full_molecule_int_to_smile







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


