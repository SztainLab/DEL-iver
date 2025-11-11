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
    parser.add_argument('bb_columns', help="a list of column names for the building block SMILES e.g. ['bba_smiles', 'bbb_smiles', ...]. These should be in the order of reactions e.g. bba, bbb, bbc, bbd") 
    parser.add_argument('output_dir', help="output directory path")
    parser.add_argument('-i', '--ids', type=str, help="name of column in input csv to be used as ids for the molecules. else, they will be automatically labelled")
    parser.add_argument('-a', '--a_lock', type=str, help="SMILES string of building block a if building block a is invariant w.r.t. molecule")
    parser.add_argument('-b', '--b_lock', type=str, help="SMILES string of building block b if building block b is invariant w.r.t. molecule")
    parser.add_argument('-c', '--c_lock', type=str, help="SMILES string of building block c if building block c is invariant w.r.t. molecule")
    parser.add_argument('-d', '--d_lock', type=str, help="SMILES string of building block d if building block d is invariant w.r.t. molecule")
    parser.add_argument('-h', '--help', action=HelpAction)
    
    args = parser.parse_args()
    
    # read the dataframe
    df = pd.read_csv(args.filename)
    
    # read the reactions as a list
    reactions = list(args.reactions)
    
    # the number of bbs that make up each full molecule
    n_bbs = len(args.reactions) + 1
    
    # specify the ids for the individual molecules either with optional argument or with generated ids
    if args.ids is None:
      ids = list(range(0, len(df)))
      df['id'] = ids
    else:
      df = df.rename(columns={args.ids: 'id'})
    
    # number of building blocks is only 3 (a-c)
    if n_bbs == 3:
      bba = {df['id'][i]: df[list(args.bb_columns)[0]][i] for i in range(len(df))}
      bbb = {df['id'][i]: df[list(args.bb_columns)[1]][i] for i in range(len(df))}
      bbc = {df['id'][i]: df[list(args.bb_columns)[2]][i] for i in range(len(df))} 
      
      # keep track of successful products after each step
      success_step1 = {}
      success_step2 = {}
      failed_1 = []
      failed_2 = [] 
      
      # run the first reaction
      for id in bba.keys():
        a = Chem.MolFromSmiles(bba[id]) if args.a_lock==None else Chem.MolFromSmiles(args.a_lock) # load the building block a from row i unless an a_lock SMILES is provided
        a.UpdatePropertyCache() # update property cache to add hydrogens
        a = Chem.AddHs(a) # add hydrogens
        b = Chem.MolFromSmiles(bbb[id]) if args.b_lock==None else Chem.MolFromSmiles(args.b_lock) # load the building block b from row i unless a b_lock SMILES is provided 
        b.UpdatePropertyCache() # update property cache to add hydrogens
        b = Chem.AddHs(b) # add explicit hydrogens
        
        for i in range(len(reactions[0])): # try all the possible reactions for step 1
          try:
            rxn = AllChem.ReactionFromSmarts(reactions[0][i]) # generate reaction from SMARTS pattern
            product = rxn.RunReactants((a,b))[0][0] # run the reaction and get the product
            product.UpdatePropertyCache() # update property cache to add hydrogens
            product = Chem.AddHs(product) # add explicit hydrogens
            success_step1[id] = Chem.MolToSmiles(product) # add product SMILES to successes from step 
            break # don't iterate through the other possible reactions if one has already worked
          
          except: # if all possible reactions for this step have failed, continue to the next set of reactants
            failed_1.append(id)
            
      print(f'success rate of step 1: {len(success_step1)/len(bba) * 100}%') # calculate success rate of step 1
      if {len(success_step1)/len(bba) * 100} == 0:
        print('no successful products from step 1')
        return
      
      # run the second reaction
      for id in success_step1.keys(): # only use the ids of the successful reactions from step 1
        prod = Chem.MolFromSmiles(success_step1[id]) # load the product from step 1
        prod.UpdatePropertyCache() # update property cache to add hydrogens
        prod = Chem.AddHs(prod) # add hydrogens
        c = Chem.MolFromSmiles(bbc[id]) if args.c_lock==None else Chem.MolFromSmiles(args.c_lock) # load the building block c from row i unless a c_lock SMILES is provided
        c.UpdatePropertyCache() # update property cache to add hydrogens
        c = Chem.AddHs(c) # add hydrogens
        
        for i in range(len(reactions[1])): # try all the possible reactions for step 2
          try: 
            rxn = AllChem.ReactionFromSmarts(reactions[1][i]) # generate reaction from SMARTS pattern
            product = rxn.RunReactants((prod,c))[0][0] # run the reaction and get the product
            product.UpdatePropertyCache() # update property cache to add hydrogens
            product = Chem.AddHs(product) # add hydrogens
            success_step2[id] = Chem.MolToSmiles(product)
            break # don't iterate through the other possible reactions if one has already worked
            
          except: # if all possible reactions for this step have failed, continue to the next set of reactants
            failed_2.append(id)
            
      print(f'success rate of step 2: {len(success_step2)/len(success_step1) * 100}%') # calculate success rate of step 2
      if (len(success_step2)/len(success_step1) * 100 == 0):
        print('no successful products from step 2')
        return
            
      print(f'total success rate: {len(success_step2)/len(bba) * 100}%')
      
      # write a csv file containing the successful full molecules 
      product_df = pd.DataFrame({'ids': list(success_step2.keys()), 'full_molecule': list(success_step2.values())})
      product_df.to_csv(f'{args.output_dir}/products.csv')
            
    # else, the number of building blocks is 4 (a-d)
    else:
      bba = {df['id'][i]: df[list(args.bb_columns)[0]][i] for i in range(len(df))}
      bbb = {df['id'][i]: df[list(args.bb_columns)[1]][i] for i in range(len(df))}
      bbc = {df['id'][i]: df[list(args.bb_columns)[2]][i] for i in range(len(df))} 
      bbd = {df['id'][i]: df[list(args.bb_columns)[2]][i] for i in range(len(df))}
      
      # keep track of successful products after each step
      success_step1 = {}
      success_step2 = {}
      success_step3 = {}
      failed_1 = []
      failed_2 = []
      failed_3 = []
      
      # run the first reaction
      for id in bba.keys():
        a = Chem.MolFromSmiles(bba[id]) if args.a_lock==None else Chem.MolFromSmiles(args.a_lock) # load the building block a from row i unless an a_lock SMILES is provided
        a.UpdatePropertyCache() # update property cache to add hydrogens
        a = Chem.AddHs(a) # add hydrogens
        b = Chem.MolFromSmiles(bbb[id]) if args.b_lock==None else Chem.MolFromSmiles(args.b_lock) # load the building block b from row i unless a b_lock SMILES is provided 
        b.UpdatePropertyCache() # update property cache to add hydrogens
        b = Chem.AddHs(b) # add explicit hydrogens
        
        for i in range(len(reactions[0])): # try all the possible reactions for step 1
          try:
            rxn = AllChem.ReactionFromSmarts(reactions[0][i]) # generate reaction from SMARTS pattern
            product = rxn.RunReactants((a,b))[0][0] # run the reaction and get the product
            product.UpdatePropertyCache() # update property cache to add hydrogens
            product = Chem.AddHs(product) # add explicit hydrogens
            success_step1[id] = Chem.MolToSmiles(product) # add product SMILES to successes from step 
            break # don't iterate through the other possible reactions if one has already worked
          
          except: # if all possible reactions for this step have failed, continue to the next set of reactants
            failed_1.append(id)
            
      print(f'success rate of step 1: {len(success_step1)/len(bba) * 100}%') # calculate success rate of step 1
      if (len(success_step1)/len(bba) * 100 == 0):
        print('no successful products from step 1')
        return
      
      # run the second reaction
      for id in success_step1.keys(): # only use the ids of the successful reactions from step 1
        prod = Chem.MolFromSmiles(success_step1[id]) # load the product from step 1
        prod.UpdatePropertyCache() # update property cache to add hydrogens
        prod = Chem.AddHs(prod) # add hydrogens
        c = Chem.MolFromSmiles(bbc[id]) if args.c_lock==None else Chem.MolFromSmiles(args.c_lock) # load the building block c from row i unless a c_lock SMILES is provided
        c.UpdatePropertyCache() # update property cache to add hydrogens
        c = Chem.AddHs(c) # add hydrogens
        
        for i in range(len(reactions[1])): # try all the possible reactions for step 2
          try: 
            rxn = AllChem.ReactionFromSmarts(reactions[1][i]) # generate reaction from SMARTS pattern
            product = rxn.RunReactants((prod,c))[0][0] # run the reaction and get the product
            product.UpdatePropertyCache() # update property cache to add hydrogens
            product = Chem.AddHs(product) # add hydrogens
            success_step2[id] = Chem.MolToSmiles(product)
            break # don't iterate through the other possible reactions if one has already worked
            
          except: # if all possible reactions for this step have failed, continue to the next set of reactants
            failed_2.append(id)
            
      print(f'success rate of step 2: {len(success_step2)/len(success_step1) * 100}%') # calculate success rate of step 2
      if (len(success_step2)/len(success_step1) * 100 == 0):
        print('no successful products from step 2')
        return
      
      # run the third reaction
      for id in success_step2.keys(): # only use the ids of the successful reactions from step 2
        prod = Chem.MolFromSmiles(success_step2[id]) # load the product from step 2
        prod.UpdatePropertyCache() # update property cache to add hydrogens
        prod = Chem.AddHs(prod) # add hydrogens
        d = Chem.MolFromSmiles(bbd[id]) if args.d_lock==None else Chem.MolFromSmiles(args.d_lock) # load the building block d from row i unless a d_lock SMILES is provided
        d.UpdatePropertyCache() # update property cache to add hydrogens
        d = Chem.AddHs(d) # add hydrogens
        
        for i in range(len(reactions[2])): # try all possible reactions for step 3
          try:
            rxn = AllChem.ReactionFromSmarts(reactions[2][i]) # generate reaction from SMARTS pattern
            product = rxn.RunReactants((prod,d))[0][0] # run the reaction and get the product
            product.UpdatePropertyCache() # update property cache to add hydrogens
            product = Chem.AddHs(product) # add hydrogens
            success_step3[id] = Chem.MolToSmiles(product)
            break # don't iterate through the other possible reactions if one has already worked
          
          except: # if all possible reactions for this step have failed, continue to the next set of reactants
            failed_3.append(id)
            
      print(f'success rate of step 3: {len(success_step3)/len(success_step2) * 100}%') # calculate success rate of step 3
      if (len(success_step3)/len(success_step2) * 100 == 0):
        print('no successful products from step 3')
        return
      
      # calculate the total success rate after all 3 reactions     
      print(f'total success rate: {len(success_step3)/len(bba) * 100}%')
      
      # write a csv file containing the successful full molecules 
      product_df = pd.DataFrame({'ids': list(success_step3.keys()), 'full_molecule': list(success_step3.values())})
      product_df.to_csv(f'{args.output_dir}/products.csv')
      
    
if __name__ == '__main__':
    main()
    