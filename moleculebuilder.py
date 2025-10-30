import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
import argparse

class HelpAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        help_text = """
Purpose: Build full molecules from building block SMILES. 

Usage: python moleculebuilder.py <filename> <reactions> <output_dir> [options]

Arguments:
  filename        - CSV file containing columns with building block SMILES (can contain additional columns as well)
  reactions LIST  - a list of lists of reactions for each step in the full molecule synthesis.
                    If there are multiple potential reactions for a step i, the possible reactions 
                    for step i will be in the inner list at reactions[i]. e.g. [[r0_0, r0_1, ...], [r1_0, r1_1, ...], ...]
  bb_columns LIST - a list of column names for the building block SMILES e.g. ['bb1_smiles', 'bb2_smiles', ...]
  output_dir      - output directory path

Options:
  -a, --a_lock STR - SMILES string of building block a if building block a is invariant w.r.t. molecule
  -b, --b_lock STR - SMILES string of building block b if building block b is invariant w.r.t. molecule
  -c, --c_lock STR - SMILES string of building block c if building block c is invariant w.r.t. molecule
  -d, --d_lock STR - SMILES string of building block d if building block d is invariant w.r.t. molecule
  -h, --help       - Show this help message
"""
        print(help_text)
        sys.exit(0)

# def generate_molecules_odlib001(reactions, bba=None, bbb=None, bbc=None, bbd=None, alock=None, block=None, clock=None, dlock=None):
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help="file path to the csv file containing columns with building block SMILES (can contain additional columns as well)")
    parser.add_argument('reactions', help="a list of lists of reactions for each step in the full molecule synthesis. If there are multiple potential reactions for a step i, the possible reactions for step i will be in the inner list at reactions[i]. e.g. [[r0_0, r0_1, ...], [r1_0, r1_1, ...], ...]")
    parser.add_argument('bb_columns', help="a list of column names for the building block SMILES e.g. ['bb1_smiles', 'bb2_smiles', ...]")
    parser.add_argument('output_dir', help="output directory path")
    parser.add_argument('-i', '--ids', type=str, help="name of column in input csv to be used as ids for the molecules. else, they will be automatically labelled")
    parser.add_argument('-a', '--a_lock', type=str, help="SMILES string of building block a if building block a is invariant w.r.t. molecule")
    parser.add_argument('-b', '--b_lock', type=str, help="SMILES string of building block b if building block b is invariant w.r.t. molecule")
    parser.add_argument('-c', '--c_lock', type=str, help="SMILES string of building block c if building block c is invariant w.r.t. molecule")
    parser.add_argument('-d', '--d_lock', type=str, help="SMILES string of building block d if building block d is invariant w.r.t. molecule")
    parser.add_argument('-h', '--help', action=HelpAction)
    
    args = parser.parse_args()
    
    # the number of bbs that make up each full molecule
    n_bbs = len(args.reactions) + 1
    
    # number of building blocks is only 3 (a-c)
    if n_bbs == 3:
      # keep track of successful products after each step
      success_step1 = {}
      success_step2 = {}
      success_step3 = {}
      failed_1 = []
      failed_2 = []
      failed_3 = []
      
      
      
      
    # else, the number of building blocks is 4 (a-d)
    else: 
    
if __name__ == '__main__':
    main()
    