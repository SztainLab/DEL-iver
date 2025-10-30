import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import argparse

from utils import *

class HelpAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        help_text = """
Purpose: Calculate ECFP4 fingerprints from a csv file containing SMILES & write to parquet file

Usage: python ecfp4_calculator.py <filename> <output_dir> [options]

Arguments:
  filename    - CSV file with a 'molecule_smiles' column (can contain additional columns as well)
  output_dir  - Output directory path

Options:
  -c, --chunk_size INT  - Chunk size for CSV reading (default: 500000)
  -s, --ecfp4_size INT  - ECFP4 fingerprint size (default: 2048)  
  -r, --remove_dy       - Remove Dy tag, replace with PEG linker
  -h, --help           - Show this help message
"""
        print(help_text)
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="file path to the train csv file, which contains a column called 'molecule_smiles'")
    parser.add_argument('output_dir', help="path to the output directory")
    parser.add_argument('-c', '--chunk_size', type=int, default=500000, help='change the chunk_size for reading in the input csv (default is 500,000)')
    parser.add_argument('-s', '--ecfp4_size', type=int, default=2048, help='change the size of ECFP4 fingerprints (default is 2048)')
    parser.add_argument('-r', '--remove_dy', action='store_true', help='if specified, the Dy tag found in some DELs will be removed and replaced with a PEG linker')
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
        
        df = chunk.copy()
        df['ecfp4_fp'] = fingerprints
        
        # Convert DataFrame to PyArrow Table and write to Parquet
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(table, root_path=args.output_dir, compression='snappy')

if __name__ == '__main__':
    main()