import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import argparse
import sys 

from utils import *

class HelpAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        help_text = """
Purpose: Calculate ECFP4 fingerprints from a csv file containing SMILES & write to parquet file

Usage: python ecfp4_calculator.py <filename> <output_dir> [options]

Arguments:
    filename    - CSV file with a 'molecule_smiles' column (can contain additional columns as well). If option -b (--bb_fingerprints) is specified, the CSV file must also contain
    columns for each building block, of the form 'building_block1_smiles', 'building_block2_smiles', etc...
    output_dir  - Output directory path

Options:
    -c, --chunk_size INT  - Chunk size for CSV reading (default: 500000)
    -s, --ecfp4_size INT  - ECFP4 fingerprint size (default: 1024)  
    -r, --remove_dy       - Remove Dy tag, replace with PEG linker
    -b, --bb_fingerprints - Calculate building block fingerprints 
    -h, --help            - Show this help message
"""
        print(help_text)
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="file path to the train csv file, which contains a column called 'molecule_smiles' and 'building_block{i}_smiles'... if -b is specified")
    parser.add_argument('output_dir', help="path to the output directory")
    parser.add_argument('-c', '--chunk_size', type=int, default=500000, help='change the chunk_size for reading in the input csv (default is 500,000)')
    parser.add_argument('-s', '--ecfp4_size', type=int, default=1024, help='change the size of ECFP4 fingerprints (default is 1024)')
    parser.add_argument('-r', '--remove_dy', action='store_true', help='if specified, the Dy tag found in some DELs will be removed and replaced with a PEG linker')
    parser.add_argument('-b', 'bb_fingerprints', action='store_true', help='if specified, fingerprints will be calculated for building blocks as well.')
    parser.add_argument('-h', '--help', action=HelpAction)
    
    args = parser.parse_args()
    
    print(f'Using {multiprocessing.cpu_count()} CPUs...')

    # Read CSV in chunks and write to Parquet in batches
    if args.chunk_size != 500000:
        chunk_size = args.chunk_size
    else:
        chunk_size = 500000
        
    reader = pd.read_csv(args.filename, chunksize=chunk_size)

    for i, chunk in tqdm(enumerate(reader)):
        print(f'Processing chunk {i}')
        if args.remove_dy == True: 
            chunk['molecule_smiles'] = chunk['molecule_smiles'].apply(replace_Dy)
            print('Dy replaced with PEG linker in SMILES')
            
        fingerprints = Parallel(n_jobs=-1)(delayed(retrieve_mol_fp)(sm, 'ECFP4', args.ecfp4_size) for sm in chunk['molecule_smiles'])
        print('Fingerprints calculated.')
        
        if args.bb_fingerprints == True: 
        # also compute the fingerprints for each building block
            bb_fingerprint_cols = []
            bb_fingerprint_colnames = []
            for column in list(chunk.columns):
                if 'building_block' in str(column):
                    bb_num =  str(column).split('block')[1]
                    bb_num = bb_num.split('_')[0]
                    bbfingerprints = Parallel(n_jobs=-1)(delayed(retrieve_mol_fp)(sm, 'ECFP4', args.ecfp4_size) for sm in chunk[column])
                    bb_fingerprint_cols.append(bb_fingerprint_cols)
                    colname = f'bb{bb_num}_ecfp4_fp'
                    bb_fingerprint_colnames.append(colname)
                
        # write to the new dataframe
        df = chunk.copy()
        # full molecule fingerprints
        df['fullmolecule_ecfp4_fp'] = fingerprints
        
        if args.bb_fingerprints == True: 
            # write the building block fingerprints
            for cname, c in zip(bb_fingerprint_colnames, bb_fingerprint_cols):
                df[cname] = c
        
        # Convert DataFrame to PyArrow Table and write to Parquet
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(table, root_path=args.output_dir, compression='snappy')

if __name__ == '__main__':
    main()