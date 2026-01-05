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
from models import *

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
    -e, --epochs INT                              - Specify the number of epochs for training (default is to scan 10, 100, 500)
    -b, --batchsize INT                           - Specify the batchsize for training (default is to scan 500, 1000, 10000)
    -l, --learningrate FLOAT                      - Specify the learning rate for training (default is to scan 0.001, 0.01, 0.1)
    -m, --models (space delimited list of models) - Specify the names of models you'd like to train (default is to train them all, except for the read count model, which can be trained by specifying the -r option)
    -f, --fullmolecule                            - Only run the full molecule models (default: False)
    -r, --readcount                               - Run the read count model (default: False)
    -h, --help                                    - Show this help message
"""
        print(help_text)
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="the parquet file output from running ecfp4_calculator.py")
    parser.add_argument('output_dir', help="path to the output directory")
    parser.add_argument('e', '--epochs', type=int, help='specify the number of epochs for training models. the default is to scan epochs=10,100,500')
    parser.add_argument('b', '--batchsize', type=int, help='specify the batchsize for training models. the default is to scan batchsize=500,1000,10000')
    parser.add_argument('l', '--learningrate', type=float, help='specify the learning rate for training. the default is to scan lr=0.001,0.01,0.1')
    parser.add_argument('m', '--models', nargs='+', help='list of models (identified by name in models.py) that you want to run. the default is to run all of them except for the read count model')
    parser.add_argument('f', '--fullmolecule', action='store_true', help='if specified, only the full molecule model will be run')
    parser.add_argument('r', '--readcount', action='store_true', help="if specified, the readcount model will be run. there must be a 'readcount' column in the file if this option is specified. the default is false.")
    parser.add_argument('-h', '--help', action=HelpAction)
    
    args = parser.parse_args()

if __name__ == '__main__':
    main()



