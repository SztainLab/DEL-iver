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

from datasets import *
from utils import *
from models import *

# NOTETOSELF!!!: Make sure the dataset version used for training is correct for each model type

class HelpAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        help_text = """
Purpose: Train models and scan hyperparameters for predicting binding probability of ligands to 

Usage: python Run_Allmodels.py <train_filename> <validation_filename> <test_filename> <output_dir> [options]

Arguments:
    train_filename      - 
    validation_filename -
    test_filename       -
    output_dir          - Output directory path

Options:
    -e, --epochs INT                              - Specify the number of epochs for training (default is to scan 10, 100, 500)
    -b, --batchsize INT                           - Specify the batchsize for training (default is to scan 500, 1000, 10000)
    -l, --learningrate FLOAT                      - Specify the learning rate for training (default is to scan 0.001, 0.01, 0.1)
    -m, --models (space delimited list of models) - Specify the names of models you'd like to train (default is to train them all, except for the read count model, which can be trained by specifying the -r option)
    -r, --readcount                               - Run the read count model (default: False)
    -h, --help                                    - Show this help message
"""
        print(help_text)
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="the parquet file output from running ecfp4_calculator.py")
    parser.add_argument('output_dir', help="path to the output directory")
    parser.add_argument('-e', '--epochs', type=int, help='specify the number of epochs for training models. the default is to scan epochs=10,100,250', default=None)
    parser.add_argument('-b', '--batchsize', type=int, help='specify the batchsize for training models. the default is to scan batchsize=500,1000,10000', default=None)
    parser.add_argument('-l', '--learningrate', type=float, help='specify the learning rate for training. the default is to scan lr=0.001,0.01,0.1', default=None)
    parser.add_argument('-m', '--models', nargs='+', help='list of models (identified by name in models.py) that you want to run. the default is to run all of them except for the read count model', default=None)
    parser.add_argument('-r', '--readcount', action='store_true', help="if specified, the readcount model will be run. there must be a 'readcount' column in the file if this option is specified. the default is false.")
    parser.add_argument('-h', '--help', action=HelpAction)
    
    args = parser.parse_args()

    # set the hyperparameter lists for training and inference
    if args.epochs == None:
        es = [10, 100, 250]
    else:
        es = [args.epochs]
        
    if args.batchsize == None:
        bss = [500, 1000, 10000]
    else:
        bss = [args.batchsize]
        
    if args.learningrate == None:
        lrs = [0.001, 0.01, 0.1]
    else:
        lrs = [args.learningrate]

    # generate list of models for training
    if args.models == None:
        models = [Simple_BuildingBlock_MLP, Simple_BuildingBlock_MLP_w_DropOut, PermutationInvariant_BuildingBlock_MLP, PermutationInvariant_BuildingBlock_MLP_V1, PermutationInvariant_BuildingBlock_MLP_V2, PermutationInvariant_BuildingBlock_MLP_V3, FullMoleculeFP_NN]
        if args.readcount == True:
            models.append(PermutationInvariant_BuildingBlock_MLP_V2_ReadCount)
    else:
        models = list(args.models)

    # load the train, test, and validation datasets

    # scan hyperparameters (epochs, batchsize, learning rate)
    for e in es:
        for bs in bss:
            for lr in lrs:
                for model in models:
                    # load the model
                    
                    
                    # train model, including validation runs and plot train/validation loss across epochs 
    
                    # run inference on test sets, and plot AUROC, mAP of model on test set

if __name__ == '__main__':
    main()



