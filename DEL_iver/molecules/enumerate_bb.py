#!/bin/python

import pickle
import argparse
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import pyarrow.compute as pc
import multiprocessing
import warnings
from tqdm import tqdm
from itertools import combinations
from DEL_iver.utils.cache import CacheNames


def _make_bb_smiles_to_id_dict(source_file, building_blocks):
    #Makes chemical ID dictioanry
    pf = pq.ParquetFile(source_file)
    # We read all columns to see where each SMILES lives
    table = pf.read(columns=building_blocks)
    
    # 1. Track which SMILES belong to which BB 'home'
    # We use a dict to ensure a SMILES is only added once
    ordered_unique_smiles = []
    seen = set()

    for block in tqdm(building_blocks,desc="Finding unique building blocks"):
        # Get unique SMILES for this specific column
        column_smiles = pc.unique(table[block]).to_pylist()
        
        # Add them to our list ONLY if we haven't seen them in a previous BB
        for s in column_smiles:
            if s not in seen:
                ordered_unique_smiles.append(s)
                seen.add(s)

    # 2. Now assign IDs based on this specific order
    # Molecules found in BB1 get low numbers (0, 1, 2...)
    # Molecules found ONLY in BB2 start after BB1's unique list ends
    smile_to_id = {smile: idx for idx, smile in enumerate(ordered_unique_smiles)}
    return smile_to_id

def _assign_id_per_row(source_file,building_blocks,smile_to_id): 
    #ASSING CHEMICAL ID PER ROW
    pf = pq.ParquetFile(source_file)
    pf= pf.read(columns=building_blocks)
    col_arrays = {}
    for block in tqdm(building_blocks, desc="Matching ID to Row"):
        encoded = pc.dictionary_encode(pf[block].cast(pa.large_string()).combine_chunks())
        block_dict = encoded.dictionary.to_pylist()
        # Map block-local dictionary to global IDs (small, only unique values)
        local_to_global = pa.array([smile_to_id[s] for s in block_dict], type=pa.int32())
        # Vectorized index remapping 
        col_arrays[f"{block}_chemical_id"] = pc.take(local_to_global, encoded.indices)
    table = pa.table(col_arrays)

    return table

def _assign_positional_id(table, building_blocks):
    """
    Adds a {block}_positional_id column for each building block.
    Each block's chemical IDs are re-enumerated independently from 0..N-1,
    so the same smile in different blocks gets different positional IDs.
    Same smile within a block always gets the same positional ID.
    """
    new_cols = {}
    offset = 0

    for block in building_blocks:
        chemical_id_col = table[f"{block}_chemical_id"].combine_chunks()

        # dictionary-encode to get a fresh 0..N-1 enumeration local to this block
        encoded = pc.dictionary_encode(chemical_id_col)
        local_ids = pc.cast(encoded.indices, pa.int32())

        # shift by cumulative offset so ranges don't overlap across blocks
        n_unique = len(encoded.dictionary)
        new_cols[f"{block}_positional_id"] = pc.add(local_ids, pa.scalar(offset, pa.int32()))
        offset += n_unique

    return pa.table({
        **{c: table[c] for c in table.schema.names},
        **new_cols
    })


def _assign_disynthon_ids(table, building_blocks):
    bb_id_cols = [f"{bb}_positional_id" for bb in building_blocks]
    combos = list(combinations(range(len(building_blocks)), 2))
    col_arrays = {}

    for combo in tqdm(combos, desc="Disynthon combos", position=0):
        i, j = combo
        combo_label = "_".join(str(x + 1) for x in combo)
        bb_a, bb_b = bb_id_cols[i], bb_id_cols[j]

        with tqdm(total=2, desc=f"  combo {combo_label}", position=1, leave=False) as inner:

            inner.set_description(f"  [{combo_label}] Cantor pairing")
            a = pc.cast(table[bb_a].combine_chunks(), pa.int64())
            b = pc.cast(table[bb_b].combine_chunks(), pa.int64())
            apb        = pc.add(a, b)
            apbp1      = pc.add(apb, pa.scalar(1, pa.int64()))
            tri        = pc.divide(pc.multiply(apb, apbp1), pa.scalar(2, pa.int64()))
            cantor_key = pc.add(tri, b)
            inner.update(1)

            inner.set_description(f"  [{combo_label}] Dictionary encoding")
            encoded = pc.dictionary_encode(cantor_key)
            col_arrays[f"disynthon_{combo_label}_id"] = pc.cast(
                encoded.indices, pa.int32()
            )
            inner.update(1)

    return pa.table({
        **{c: table[c] for c in table.schema.names},
        **col_arrays
    })



def enumerate_building_blocks(ddr):
    """
    To do: Add notes

    """

    source_file=ddr.source_file
    building_blocks=ddr.building_blocks

    output_path = ddr.cache.get_output_path(CacheNames.BB_DICTIONARIES, "main")

    if os.path.exists(output_path):
        warnings.warn(f"BB enumaration found in cache no further work needed",UserWarning)

    else:
        smile_to_id=_make_bb_smiles_to_id_dict(source_file,building_blocks)
        id_to_smile = {idx: smile for smile, idx in smile_to_id.items()}
        id_to_smile_table = pa.table({
            "id": list(id_to_smile.keys()),
            "smiles": list(id_to_smile.values())
        })

        pq.write_table(
            id_to_smile_table,
            ddr.cache.get_output_path(CacheNames.BB_DICTIONARIES, "id_to_smiles")
        )

        table=_assign_id_per_row(source_file,building_blocks,smile_to_id)
        table = _assign_positional_id(table, building_blocks)
        table=_assign_disynthon_ids(table,building_blocks)
        pq.write_table(table, output_path) 



