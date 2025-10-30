#!/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm

# function to generate molecules from building blocks
# reactions should be specified as a lists of lists where reactions[0] is a list that contains the possible reactions for step1, in the order they should be tried
# specific to odlib001.. 
def generate_molecules_odlib014(reactions, bba=None, bbb=None, bbc=None, bbd=None, alock=None, block=None, clock=None, dlock=None):
    success_step1 = {} # keep track of successful products at each step
    success_step2 = {} 
    success_step3 = {}
    failed_1 = []
    failed_2 = []
    failed_3 = []
    
    # if there are three building blocks (aside from building block a)
    if bbd is not None: # three building blocks (b,c,d) 
        
        # step1!
        for id in bbb.keys(): # using bbb here because alock is often not None
            a = Chem.MolFromSmiles(bba[id]) if alock==None else Chem.MolFromSmiles(alock) # use the building block a unless alock is specified
            a.UpdatePropertyCache() # update property cache
            a = Chem.AddHs(a) # add explicit hydrogens
            b = Chem.MolFromSmiles(bbb[id]) if block==None else Chem.MolFromSmiles(block) # use the building block b unless block is specified
            # b.UpdatePropertyCache() # update property cache
            # b = Chem.AddHs(b) # add explicit hydrogens
            worked = False
            for i in range(len(reactions[0])): # try all the possible reactions for the given step (step1 here)
                try:
                    rxn = AllChem.ReactionFromSmarts(reactions[0][i]) # generate reaction from SMARTS pattern
                    product = rxn.RunReactants((a,b))[0][0] # extract the reaction
                    product.UpdatePropertyCache() # update property cache
                    product = Chem.AddHs(product) # add explicit hydrogens
                    success_step1[id] = Chem.MolToSmiles(product) # add product smiles to success_step1
                    worked = True
                    break # don't try the other reactions if one has worked
                except: # if all possible reactions for this step have failed, continue to the next reaction
                    continue
            if worked == False:
                failed_1.append(id)
                
        print(f'success of step 1: {len(success_step1)/len(bbb) * 100}%') # calculate success of step1 based on total number of molecules
        if {len(success_step1)/len(bbb) * 100} == 0:
            return
        
        # step2!
        for id in success_step1.keys(): # use the products from step1 as one reactant for step2 
            prod = Chem.MolFromSmiles(success_step1[id])
            # prod.UpdatePropertyCache() # update property cache .. doing this again is probably overkill but don't think it can hurt anything? 
            # prod = Chem.AddHs(prod) # add explicit hydrogens .. ^^^
            c = Chem.MolFromSmiles(bbc[id]) if clock==None else Chem.MolFromSmiles(clock) # use bbc unless clock is specified
            c.UpdatePropertyCache() # update property cache
            c = Chem.AddHs(c) # add explicit hydrogens
            worked = False
            for i in range(len(reactions[1])):
                try:
                    rxn = AllChem.ReactionFromSmarts(reactions[1][i])
                    product = rxn.RunReactants((prod,c))[0][0]
                    # product.UpdatePropertyCache() # update property cache
                    # product = Chem.AddHs(product) # add explicit hydrogens
                    success_step2[id] = Chem.MolToSmiles(product)
                    worked = True
                    break # don't try the other reactions if one has worked already
                except:
                    continue
            if worked == False:
                failed_2.append(id)
                
                
        print(f'success of step 2: {len(success_step2)/len(success_step1) * 100}%') # calculate success of step2 based on total number of molecules generated from step1
        if  {len(success_step2)/len(success_step1) * 100} == 0:
            return 
        
        # step3!
        for id in success_step2.keys():
            prod = Chem.MolFromSmiles(success_step2[id])
            prod.UpdatePropertyCache() # prob overkill again
            # prod = Chem.AddHs(prod)
            d = Chem.MolFromSmiles(bbd[id]) if dlock==None else Chem.MolFromSmiles(dlock) # use bbd unless dlock is specified
            d.UpdatePropertyCache() # update property cache
            # d = Chem.AddHs(d) # add explicit hydrogens
            worked = False
            for i in range(len(reactions[2])):
                try:
                    rxn = AllChem.ReactionFromSmarts(reactions[2][i])
                    product = rxn.RunReactants((prod,d))[0][0]
                    product.UpdatePropertyCache() # update property cache
                    product = Chem.AddHs(product)

                    product = Chem.SanitizeMol(product)
                    for a in product.getAtoms():
                        if a.GetNumImplicitHs():
                            a.SetNumRadicalElectrons(a.GetNumImplicitHs())
                            a.SetNoImplicit(True)
                            a.UpdatePropertyCache()
                    product = Chem.RemoveAllHs(product)

                    success_step3[id] = Chem.MolToSmiles(product)
                    worked = True
                    break # don't try the other reactions if one has worked already
                except:
                    continue
            if worked == False:
                failed_3.append(id)
                
        print(f'success of step 3: {len(success_step3)/len(success_step2) * 100}%') # success based on molecules generated from step 2
        print('reactions complete after step 3.')
        
        if {len(success_step3)/len(success_step2) * 100} == 0:
            return 
        
        print(f'total success: {len(success_step3)/len(bbb) * 100}%') # success calculated based on all the molecules in the set
        return success_step1, success_step2, success_step3, failed_1, failed_2, failed_3
        
    # if the library only contains 2 building blocks aside from a: 
    else: # two building blocks (b,c)
        # step1!
        for id in bbb.keys():
            a = Chem.MolFromSmiles(bba[id]) if alock==None else Chem.MolFromSmiles(alock) # use bba unless alock is specified
            a.UpdatePropertyCache() # update property cache
            # a = Chem.AddHs(a) # add explicit hydrogens
            b = Chem.MolFromSmiles(bbb[id]) if block==None else Chem.MolFromSmiles(block) # use bbb unless block is specified
            b.UpdatePropertyCache() # update property cache
            # b = Chem.AddHs(b) # add explicit hydrogens
            worked = False
            for i in range(len(reactions[0])):
                try:
                    rxn = AllChem.ReactionFromSmarts(reactions[0][i])
                    product = rxn.RunReactants((a,b))[0][0]
                    product.UpdatePropertyCache() # update property cache
                    product = Chem.AddHs(product) # add explicit hydrogens
                    success_step1[id] = Chem.MolToSmiles(product)
                    worked = True
                    break
                except:
                    continue
            if worked == False:
                failed_1.append(id)
                
        print(f'success of step 1: {len(success_step1)/len(bbb) * 100}%') # success of step1 
        if  {len(success_step1)/len(bbb) * 100} == 0:
            return
        
        # step2!
        for id in success_step1.keys():
            prod = Chem.MolFromSmiles(success_step1[id])
            prod.UpdatePropertyCache() # prob overkill
            prod = Chem.AddHs(prod) # prob overkill
            c = Chem.MolFromSmiles(bbc[id]) if clock==None else Chem.MolFromSmiles(clock) # use bbc unless clock is specified
            c.UpdatePropertyCache() # update property cache
            # c = Chem.AddHs(c) # add explicit hydrogens
            worked = False
            for i in range(len(reactions[1])):
                try:
                    rxn = AllChem.ReactionFromSmarts(reactions[1][i])
                    product = rxn.RunReactants((prod,c))[0][0]
                    product.UpdatePropertyCache() # update property cache
                    product = Chem.AddHs(product) # add explicit hydrogens
                    success_step2[id] = Chem.MolToSmiles(product)
                    worked = True
                    break
                except:
                    continue
            if worked == False:
                failed_2.append(id)
                
        print(f'success of step 2: {len(success_step2)/len(success_step1) * 100}%')
        print('reactions complete after step 2.')
        
        if {len(success_step2)/len(success_step1) * 100} == 0:
            return
        
        print(f'total success: {len(success_step2)/len(bbb) * 100}%')
    
        return success_step1, success_step2, failed_1, failed_2
    
    

# molecule generation
data = 'Truttman_092724_analysis_ready_files/ODLIB014/Truttman-092724_ODLIB014_XCAMP01034_CUBE_pdickson.16299615.notenumerated.csv'
ground_truth = 'exemplar_fullmolecules/ODLIB014_Instance_Exemplars_withIDs.csv'

df = pd.read_csv(data)
true_df = pd.read_csv(ground_truth)

df['id'] = df.index
print(list(df.columns))
print(list(true_df.columns))

reactions = [['[CX3:3]([#1:4])(=O)[#6,#1:5].[NX3!H0;!$([N][!#6]);!$([NX3][A]=,#[A$([!#6])]):1][#6:2]>>[N:1]([#6:2])[#6:3]([#1:4])[*:5]'], ['[NX3!H0;!$([N][!#6]);!$([NX3][A]=,#[A$([!#6])]):1][#6:2].[CX3:3]([#1:4])(=O)[#6,#1:5]>>[N:1]([#6:2])[#6:3]([#1:4])[*:5]'], ['[CX3:3]([OD1h1])(=O)[#6:4].[NX3!H0;!$([N][!#6]);!$([NX3][A]=,#[A$([!#6])]):1][#6:2]>>[#6:4][C:3](=O)[N:1][#6:2]']]
a_lock = 'C=O' # molecule to be used for all bba cycles (as specified in SMARTS library excel file)


bba = {} 
bbb = {df['id'][i]: df['BSMILES'][i] for i in range(df.shape[0])}
bbc = {df['id'][i]: df['CSMILES'][i] for i in range(df.shape[0])} 
bbd = {df['id'][i]: df['DSMILES'][i] for i in range(df.shape[0])}

success1, success2, success3, failed1, failed2, failed3 = generate_molecules_odlib014(reactions, bba, bbb, bbc, bbd, alock=a_lock)

print(len(failed1), len(failed2), len(failed3))




# write csv file and analyze against ground truth
gen_df = pd.DataFrame(list(success3.items()), columns=["id", "gen_mole"])

gen_df = pd.merge(gen_df, df, on="id", how="left")

true_df.rename(columns={'BBB ID': 'BBB', 'BBC ID': 'BBC', 'BBD ID': 'BBD'}, inplace=True)

final_df = pd.merge(gen_df, true_df, on=['BBB', 'BBC', 'BBD'], how="left")

final_df.to_csv('ODLIB014_fullmolecule_generation.csv')

final_df.head()

filtered_df = final_df[final_df["SMILES"].notna()]

print(filtered_df.shape)

gen_moles = list(filtered_df['gen_mole'])
real_moles = list(filtered_df['SMILES'])

for i in range(len(gen_moles)):
    print(gen_moles[i], real_moles[i])