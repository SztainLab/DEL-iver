#!/bin/python

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
Purpose: Train models 

Usage: python Run_Allmodels.py <filename> <output_dir> [options]

Arguments:
    filename    - the parquet file output from running ecfp4_calculator.py
    output_dir  - Output directory path

Options:
    -e, --epochs INT         - Specify the number of epochs for training (default is to scan 10, 100, 500)
    -b, --batchsize INT      - Specify the batchsize for training (default is to scan 500, 1000, 10000)
    -l, --learningrate FLOAT - Specify the (default is to scan 0.001, 0.01, 0.1)
    -m, --models STR         - Specify the names of models you'd like to train (default is to train them all, except for the read count model, which can be trained by specifying the -r option)
    -f, --fullmolecule       - Only run the full molecule models (default: False)
    -r, --readcount          - Run the read count model (default: False)
    -h, --help               - Show this help message
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

if __name__ == '__main__':
    main()




