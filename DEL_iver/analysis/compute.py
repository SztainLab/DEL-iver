from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from DEL_iver.utils.cache import CacheManager, CacheNames
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, GraphDescriptors, QED
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem.MolStandardize import rdMolStandardize

def _load_tables(ddr, building_blocks,split_col=None): #!april7 correct
    bb_table = ddr.cache.get_path(
        CacheNames.BB_DICTIONARIES,
        filename=f"{CacheNames.BB_DICTIONARIES.value}.{ddr.source_file.stem}.parquet"
    )
    if not ddr.cache.is_cached(bb_table):
        raise RuntimeError("BB dictionaries not found. Run generate_bb_dictionaries(ddr) first.")

    columns = building_blocks + [ddr.label]

    if split_col is not None:
        columns.append(split_col)

    bb_table = pq.read_table(bb_table)
    source_table = pq.read_table(ddr.source_file, columns=columns)

    for col in bb_table.schema.names:
        source_table = source_table.append_column(col, bb_table[col])

    return source_table, bb_table


def _get_global_totals(source_table, label_col): 
    total_hits = pc.sum(pc.equal(source_table[label_col], 1)).as_py()
    total_nonhits = pc.sum(pc.equal(source_table[label_col], 0)).as_py()
    return total_hits, total_nonhits


def _count_hits_and_total(table, group_col, label_col,extra_col=None):  
    if extra_col:
        group_col=[group_col+extra_col]
        
    hits = (
        table.filter(pc.equal(table[label_col], 1))
        .group_by(group_col)
        .aggregate([(label_col, "count")])
        .rename_columns(["id", "nhits"])
    )
    total = (
        table.group_by(group_col)
        .aggregate([(label_col, "count")])
        .rename_columns(["id", "ntotal"])
    )
    return total.join(hits, "id", join_type="left outer")



def _aggregate_bb_counts_across_positions(source_table, bb_id_cols, building_blocks, label_col):
    chemical_id_cols = [f"{bb}_chemical_id" for bb in building_blocks]
    chunks = []
    for bb_id_col, chem_id_col in zip(bb_id_cols, chemical_id_cols):
        chunk = source_table.select([bb_id_col, chem_id_col, label_col]).rename_columns(["positional_id", "chemical_id", label_col])
        chunk = chunk.append_column("origin", pa.array([bb_id_col] * len(chunk)))
        chunks.append(chunk)
    stacked = pa.concat_tables(chunks)

    total    = stacked.group_by(["positional_id", "chemical_id", "origin"]).aggregate([(label_col, "count")]).rename_columns(["positional_id", "chemical_id", "origin", "ntotal"])
    filtered = stacked.filter(pc.equal(stacked[label_col], 1))
    hits     = filtered.group_by(["positional_id", "chemical_id", "origin"]).aggregate([(label_col, "count")]).rename_columns(["positional_id", "chemical_id", "origin", "nhits"])
    counts   = total.join(hits, keys=["positional_id", "chemical_id", "origin"], join_type="left outer")

    return counts


def _apply_enrichment(stats, total_hits, total_nonhits, method, min_occurrences): #TODO: remove not needed input flags here
    # --- clean nhits (nulls → 0) and cast both to int32 ---
    nhits_is_null_mask = pc.is_null(stats["nhits"])
    nhits_int  = pc.if_else(nhits_is_null_mask, 0, stats["nhits"]).cast(pa.int32())
    ntotal_int = stats["ntotal"].cast(pa.int32())
    stats = stats.set_column(stats.schema.get_field_index("nhits"),  "nhits",  nhits_int)
    stats = stats.set_column(stats.schema.get_field_index("ntotal"), "ntotal", ntotal_int)

    if min_occurrences > 0:
        stats = stats.filter(pc.greater_equal(stats["ntotal"], min_occurrences))

    nhits  = stats["nhits"]
    ntotal = stats["ntotal"]

    # nnonhits and pbind (raw, no smoothing — position-dependent)
    nnonhits = pc.subtract(ntotal, nhits)
    stats = stats.append_column("nnonhits", nnonhits)
    pbind = pc.divide(nhits.cast(pa.float64()), ntotal.cast(pa.float64()))
    stats = stats.append_column("pbind", pbind)

    # --- per-origin totals (vectorized) ---
    origin_totals = (
        stats
        .group_by("origin")
        .aggregate([("nhits", "sum"), ("ntotal", "sum")])
        .rename_columns(["origin", "origin_total_hits", "origin_total_compounds"])
    )
    stats = stats.join(origin_totals, keys="origin", join_type="left outer")

    # --- float64 for numerical stability ---
    nhits_f          = stats["nhits"].cast(pa.float64())
    ntotal_f         = stats["ntotal"].cast(pa.float64())
    origin_hits      = stats["origin_total_hits"].cast(pa.float64())
    origin_compounds = stats["origin_total_compounds"].cast(pa.float64())

    # --- p_scaffold and p_origin with smoothing ---
    if method.lower() == "laplace":
        p_scaffold = pc.divide(pc.add(nhits_f, 1.0), pc.add(ntotal_f, 2.0))
        p_origin   = pc.divide(pc.add(origin_hits, 1.0), pc.add(origin_compounds, 2.0))
    elif method.lower() == "epsilon":
        eps = 1e-6
        p_scaffold = pc.add(pc.divide(nhits_f, ntotal_f), eps)
        p_origin   = pc.add(pc.divide(origin_hits, origin_compounds), eps)
    else:
        raise ValueError("Method must be 'laplace' or 'epsilon'")

    enrichment = pc.divide(p_scaffold, p_origin)

    stats = stats.append_column("p_scaffold", p_scaffold)
    stats = stats.append_column("p_origin",   p_origin)
    stats = stats.append_column("enrichment", enrichment)

    return stats



def _count_hits_and_total_disynthon(table, dis_col, bb_positional_id_cols, bb_chemical_id_cols, label_col):
    group_cols = [dis_col] + bb_positional_id_cols + bb_chemical_id_cols

    hits = (
        table.filter(pc.equal(table[label_col], 1))
        .group_by(group_cols)
        .aggregate([(label_col, "count")])
        .rename_columns(group_cols + ["nhits"])
    )
    total = (
        table.group_by(group_cols)
        .aggregate([(label_col, "count")])
        .rename_columns(group_cols + ["ntotal"])
    )
    result = total.join(hits, group_cols, join_type="left outer")

    result = result.rename_columns(
        ["positional_id" if c == dis_col else c for c in result.schema.names]
    )
    result = result.append_column(
        "origin", pa.array([dis_col] * len(result))
    )
    return result


def _compute_bb_enrichment(source_table, bb_id_cols, building_blocks, label_col, total_hits, total_nonhits, method, min_occurrences, ignore_position):
    

    counts = _aggregate_bb_counts_across_positions(source_table, bb_id_cols, building_blocks, label_col)
    stats = _apply_enrichment(counts, total_hits, total_nonhits, method, min_occurrences) #!ALL GOOD UP TO HERE IS CORRECT
    stats  = stats.append_column("type", pa.array(["building_block"] * len(stats)))
    return stats

def _compute_disynthon_enrichment(source_table, dis_col, relevant_bb_positional_cols, relevant_bb_chemical_cols, label_col, total_hits, total_nonhits, method, min_occurrences):
    counts = _count_hits_and_total_disynthon(source_table, dis_col, relevant_bb_positional_cols, relevant_bb_chemical_cols, label_col)
    stats  = _apply_enrichment(counts, total_hits, total_nonhits, method, min_occurrences)
    stats  = stats.append_column("type", pa.array(["disynthon"] * len(stats)))
    return stats

def _write_output(ddr, table):
    output_path = ddr.cache.get_path(
        CacheNames.COMPUTE,
        filename=f"{CacheNames.COMPUTE.value}.{ddr.source_file.stem}.parquet"
    )
    pq.write_table(table, output_path)
    print(f"Written to {output_path}")


def compute_pbind_and_enrichment(ddr, method='laplace', min_occurrences=0,ignore_position: bool =False,draw: bool = False):

    building_blocks = ddr.building_blocks 
    source_table, bb_table = _load_tables(ddr, building_blocks)
    disynthon_cols = ddr.disynthons 
    bb_id_cols = [f"{bb}_positional_id" for bb in building_blocks]


    total_hits, total_nonhits = _get_global_totals(source_table, ddr.label) 
    bb_stats = _compute_bb_enrichment( 
        source_table, 
        bb_id_cols, 
        building_blocks,
        ddr.label, 
        total_hits, 
        total_nonhits, 
        method, 
        min_occurrences,
        ignore_position
    )




    bb_positional_id_cols = [f"{bb}_positional_id" for bb in building_blocks]
    bb_chemical_id_cols = [f"{bb}_chemical_id" for bb in building_blocks]

    disynthon_stats = {}
    for dis_col in tqdm(ddr.disynthons, desc="Computing Enrichment and Pbind"):
        combo_indices               = [int(i) - 1 for i in dis_col.replace("disynthon_", "").replace("_id", "").split("_")]
        relevant_bb_positional_cols = [bb_positional_id_cols[i] for i in combo_indices]
        relevant_bb_chemical_cols   = [bb_chemical_id_cols[i]   for i in combo_indices]
        disynthon_stats[dis_col]    = _compute_disynthon_enrichment(
            source_table,
            dis_col,
            relevant_bb_positional_cols,
            relevant_bb_chemical_cols,
            ddr.label,
            total_hits,
            total_nonhits,
            method,
            min_occurrences
        )

    final_table = pa.concat_tables([bb_stats] + list(disynthon_stats.values()), promote_options="default")


    _write_output(ddr, final_table)

    return bb_stats, disynthon_stats




def _compute_descriptors_single(smi: str) -> dict:
    """Worker function — must be top-level for multiprocessing pickle."""
    try:
        if not smi:
            return {"smiles": smi, "_valid": False}

        mol = Chem.MolFromSmiles(smi)  # <-- this was missing

        if mol is None:
            return {"smiles": smi, "_valid": False}

        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)

        if mol is None:
            return {"smiles": smi, "_valid": False}

        Chem.SanitizeMol(mol)
        Chem.FastFindRings(mol)

    except Exception:
        return {"smiles": smi, "_valid": False}

    row = {"smiles": smi, "_valid": True}
    # ... rest of descriptors

    # --- Lipinski / drug-likeness ---
    row["mol_wt"]                     = Descriptors.MolWt(mol)
    row["exact_mol_wt"]               = Descriptors.ExactMolWt(mol)
    row["logp"]                       = Descriptors.MolLogP(mol)
    row["tpsa"]                       = CalcTPSA(mol)
    row["hbd"]                        = rdMolDescriptors.CalcNumHBD(mol)
    row["hba"]                        = rdMolDescriptors.CalcNumHBA(mol)
    row["rotatable_bonds"]            = rdMolDescriptors.CalcNumRotatableBonds(mol)
    row["heavy_atom_count"]           = mol.GetNumHeavyAtoms()
    row["num_heteroatoms"]            = rdMolDescriptors.CalcNumHeteroatoms(mol)
    row["fraction_csp3"]              = rdMolDescriptors.CalcFractionCSP3(mol)
    row["qed"]                        = QED.qed(mol)

    # --- Lipinski rule of 5 flags ---
    row["ro5_mw"]                     = row["mol_wt"] <= 500
    row["ro5_logp"]                   = row["logp"] <= 5
    row["ro5_hbd"]                    = row["hbd"] <= 5
    row["ro5_hba"]                    = row["hba"] <= 10
    row["passes_ro5"]                 = all([
                                          row["ro5_mw"], row["ro5_logp"],
                                          row["ro5_hbd"], row["ro5_hba"]
                                        ])

    # --- Electronic ---
    row["max_partial_charge"]         = Descriptors.MaxPartialCharge(mol)
    row["min_partial_charge"]         = Descriptors.MinPartialCharge(mol)
    row["max_abs_partial_charge"]     = Descriptors.MaxAbsPartialCharge(mol)
    row["min_abs_partial_charge"]     = Descriptors.MinAbsPartialCharge(mol)
    row["num_radical_electrons"]      = Descriptors.NumRadicalElectrons(mol)
    row["num_valence_electrons"]      = Descriptors.NumValenceElectrons(mol)

    # --- Structural / ring ---
    row["num_rings"]                  = rdMolDescriptors.CalcNumRings(mol)
    row["num_aromatic_rings"]         = rdMolDescriptors.CalcNumAromaticRings(mol)
    row["num_aliphatic_rings"]        = rdMolDescriptors.CalcNumAliphaticRings(mol)
    row["num_saturated_rings"]        = rdMolDescriptors.CalcNumSaturatedRings(mol)
    row["num_aromatic_carbocycles"]   = rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
    row["num_aromatic_heterocycles"]  = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    row["num_stereocenters"]          = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    row["num_unspecified_stereo"]     = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    row["num_spiro_atoms"]            = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    row["num_bridgehead_atoms"]       = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    row["num_amide_bonds"]            = rdMolDescriptors.CalcNumAmideBonds(mol)

    # --- Atom composition ---
    atom_counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        atom_counts[sym] = atom_counts.get(sym, 0) + 1
    for element in ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]:
        row[f"count_{element}"]       = atom_counts.get(element, 0)

    # --- Bond types ---
    row["num_atoms"]                  = mol.GetNumAtoms()
    row["num_bonds"]                  = mol.GetNumBonds()
    row["num_heavy_bonds"]            = sum(
                                          1 for b in mol.GetBonds()
                                          if b.GetBeginAtom().GetAtomicNum() > 1
                                          and b.GetEndAtom().GetAtomicNum() > 1
                                        )
    row["num_aromatic_bonds"]         = sum(
                                          1 for b in mol.GetBonds()
                                          if b.GetIsAromatic()
                                        )
    row["num_single_bonds"]           = sum(
                                          1 for b in mol.GetBonds()
                                          if b.GetBondTypeAsDouble() == 1.0
                                          and not b.GetIsAromatic()
                                        )
    row["num_double_bonds"]           = sum(
                                          1 for b in mol.GetBonds()
                                          if b.GetBondTypeAsDouble() == 2.0
                                        )
    row["num_triple_bonds"]           = sum(
                                          1 for b in mol.GetBonds()
                                          if b.GetBondTypeAsDouble() == 3.0
                                        )

    return row


def compute_chemical_descriptors(ddr,
                                 n_jobs: int = None,
                                 chunksize: int = 100,
                                 multiprocess: bool = False) -> pa.Table:
    """
    Computes a wide array of RDKit chemical descriptors for all unique SMILES
    in the bb_dictionaries id_to_smiles parquet.

    Args:
        ddr          : data context object
        n_jobs       : number of worker processes (only used if multiprocess=True)
        chunksize    : SMILES per worker task (only used if multiprocess=True)
        multiprocess : if True, use multiprocessing (requires /dev/shm space).
                       Default False — single threaded is fast enough for typical
                       BB dictionaries and avoids shared memory issues on clusters.
    """
    bb_dict_path = ddr.cache.get_path(
        CacheNames.BB_DICTIONARIES,
        filename=f"{CacheNames.BB_DICTIONARIES.value}.{ddr.source_file.stem}.parquet"
    )
    id_to_smiles_path = bb_dict_path.with_name(
        bb_dict_path.stem + "_id_to_smiles.parquet"
    )

    if not id_to_smiles_path.exists():
        raise FileNotFoundError(f"id_to_smiles parquet not found at: {id_to_smiles_path}")

    id_smiles_table = pq.read_table(id_to_smiles_path)
    ids    = id_smiles_table["id"].to_pylist()
    smiles = id_smiles_table["smiles"].to_pylist()

    print(f"Loaded {len(smiles):,} unique SMILES from: {id_to_smiles_path.name}")
    #TODO: ADD CUSTOM TDQM FOR STEP 1
    if multiprocess:
        if n_jobs is None:
            n_jobs = max(1, min(cpu_count() - 1, len(smiles) // chunksize, 16))
        print(f"Computing descriptors using {n_jobs} workers (chunksize={chunksize})...")
        with Pool(processes=n_jobs) as pool:
            records = list(tqdm(
                pool.imap(_compute_descriptors_single, smiles, chunksize=chunksize),
                total=len(smiles),
                desc="Computing descriptors"
            ))
    else:
        print(f"Computing descriptors single-threaded...")
        records = [
            _compute_descriptors_single(smi)
            for smi in tqdm(smiles, desc="Computing descriptors")
        ]

    # --- Convert to PyArrow Table ---
    all_keys = [k for k in records[0].keys() if k not in ("smiles", "_valid")]
    col_arrays = {"id": pa.array(ids, type=pa.int64())}

    for key in tqdm(all_keys, desc="Building PyArrow columns"):
        raw = [r.get(key, None) if r.get("_valid", False) else None for r in records]
        first = next((v for v in raw if v is not None), None)
        if isinstance(first, bool):
            arr = pa.array(raw, type=pa.bool_())
        elif isinstance(first, int):
            arr = pa.array(raw, type=pa.int32())
        elif isinstance(first, float):
            arr = pa.array(raw, type=pa.float32())
        else:
            arr = pa.array(raw)
        col_arrays[key] = arr

    col_arrays["smiles"] = pa.array(smiles, type=pa.string())
    table = pa.table(col_arrays)

    invalid_count = int(pc.sum(pc.is_null(table["mol_wt"])).as_py())
    print(f"Done. {len(table) - invalid_count:,} valid / {invalid_count:,} invalid "
          f"out of {len(smiles):,} total.")

    #TODO CHANGE TO COMPUTE PATH
    output_path = bb_dict_path.with_name(
        bb_dict_path.stem + "_descriptors.parquet"
    )
    pq.write_table(table, output_path)
    print(f"Saved descriptors to: {output_path}")
    print(table)
    return table




import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

def find_best_bb(ddr, n, min_occurrences=0, sort_by="pbind", exclude: list = None):
    if sort_by not in ("pbind", "enrichment"):
        raise ValueError(f"sort_by must be 'pbind' or 'enrichment', got '{sort_by}'")

    enrichment_path = ddr.cache.get_path(
        CacheNames.COMPUTE,
        filename=f"{CacheNames.COMPUTE.value}.{ddr.source_file.stem}.parquet"
    )
    enrichment = pq.read_table(enrichment_path)

    id_to_smile_path = ddr.cache.get_path(
        CacheNames.BB_DICTIONARIES,
        filename=f"{CacheNames.BB_DICTIONARIES.value}.{ddr.source_file.stem}_id_to_smiles.parquet"
    )
    id_to_smile_table = pq.read_table(id_to_smile_path)
    id_to_smile = {
        int(row["id"]): row["smiles"]
        for row in id_to_smile_table.to_pylist()
    }

    if "type" in enrichment.schema.names:
        bb_stats = enrichment.filter(pc.equal(enrichment["type"], "building_block"))
    else:
        bb_stats = enrichment

    exclude = [exclude] if isinstance(exclude, str) else (exclude or [])
    building_blocks = ddr.building_blocks
    bb_positional_id_cols = [f"{bb}_positional_id" for bb in building_blocks]

    mapped_exclude = []
    for item in exclude:
        if isinstance(item, int):
            if 1 <= item <= len(bb_positional_id_cols):
                mapped_exclude.append(bb_positional_id_cols[item - 1])
            else:
                raise IndexError(f"Exclude index {item} out of bounds for building_blocks of size {len(building_blocks)}.")

    if mapped_exclude and "origin" in bb_stats.schema.names:
        keep_mask = pc.invert(pc.is_in(bb_stats["origin"], value_set=pa.array(mapped_exclude)))
        bb_stats = bb_stats.filter(keep_mask)

    if sort_by not in bb_stats.schema.names:
        raise ValueError(f"Column '{sort_by}' not found. Available: {bb_stats.schema.names}")

    bb_stats_sorted = bb_stats.sort_by([(sort_by, "descending")])
    all_rows = bb_stats_sorted.to_pylist()

    # single pass: collect rows in rank order, stop once n unique chemical_ids seen
    # if the same chemical_id appears again before the cutoff, include that row too
    seen_set = set()
    selected_rows = []
    for row in all_rows:
        chem_id = row["chemical_id"]
        if chem_id in seen_set:
            selected_rows.append(row)  # repeat within the window
        else:
            seen_set.add(chem_id)
            selected_rows.append(row)  # first occurrence
            if len(seen_set) == n:
                break

    top_n = pa.Table.from_pylist(selected_rows, schema=bb_stats_sorted.schema)

    smiles_list = []
    for i in top_n["chemical_id"].to_pylist():
        if i is None:
            raise ValueError("Null chemical_id found in top_n — this should not happen.")
        if int(i) not in id_to_smile:
            raise KeyError(f"chemical_id {i} not found in id_to_smile — dictionary may be out of sync.")
        smiles_list.append(id_to_smile[int(i)])

    top_n = top_n.append_column("smiles", pa.array(smiles_list))

    for row in top_n.to_pylist():
        origin  = row.get("origin", "Pooled")
        smile   = row.get("smiles")
        score   = row.get(sort_by, 0.0)
        pos_id  = row["positional_id"]
        chem_id = row["chemical_id"]
        print(f"positional_id: {pos_id} | chemical_id: {chem_id} | Origin: {origin} | {sort_by}: {score:.4f} | SMILES: {smile}")

    return top_n



import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


#TODO: FIX ONLY FIRST DISINTHON GETTING WHITE BACKGROUND EITHER ALL OR NONE
def find_best_disynthon(ddr, n, min_occurrences=0, sort_by="pbind", exclude=None):
    if sort_by not in ("pbind", "enrichment"):
        raise ValueError(f"sort_by must be 'pbind' or 'enrichment', got '{sort_by}'")

    # --- Standardize exclude input ---
    if exclude is None:
        exclude = []
    elif isinstance(exclude, (str, tuple)):
        exclude = [exclude]

    mapped_exclude = []
    for item in exclude:
        if isinstance(item, tuple):
            suffix = "_".join(str(idx) for idx in item)
            mapped_exclude.append(f"disynthon_{suffix}_id")
        else:
            mapped_exclude.append(str(item))

    # --- Load enrichment table ---
    enrichment_path = ddr.cache.get_path(
        CacheNames.COMPUTE,
        filename=f"{CacheNames.COMPUTE.value}.{ddr.source_file.stem}.parquet"
    )
    enrichment = pq.read_table(enrichment_path)

    # --- Load id-to-smiles dictionary ---
    id_to_smile_path = ddr.cache.get_path(
        CacheNames.BB_DICTIONARIES,
        filename=f"{CacheNames.BB_DICTIONARIES.value}.{ddr.source_file.stem}_id_to_smiles.parquet"
    )
    id_to_smile_table = pq.read_table(id_to_smile_path)
    id_to_smile = {int(r["id"]): r["smiles"] for r in id_to_smile_table.to_pylist()}

    # --- Filter to only disynthon rows (origin starts with "disynthon_") ---
    if "origin" in enrichment.schema.names:
        dis_stats = enrichment.filter(
            pc.starts_with(pc.cast(enrichment["origin"], pa.string()), "disynthon_")
        )
    else:
        dis_stats = enrichment

    # --- Apply min_occurrences filter ---
    if min_occurrences > 0:
        dis_stats = dis_stats.filter(pc.greater_equal(dis_stats["ntotal"], min_occurrences))

    # --- Apply exclude filter by origin ---
    if mapped_exclude and dis_stats.num_rows > 0:
        keep_mask = pc.invert(pc.is_in(dis_stats["origin"], value_set=pa.array(mapped_exclude)))
        dis_stats = dis_stats.filter(keep_mask)

    if dis_stats.num_rows == 0:
        print("No disynthon data found after filtering.")
        return dis_stats

    # --- Sort and slice ---
    top_n = dis_stats.sort_by([(sort_by, "descending")]).slice(0, n)

    # --- Construct SMILES from chemical_id columns ---
    # e.g. buildingblock1_smiles_chemical_id, buildingblock2_smiles_chemical_id
    bb_chemical_cols = sorted([
        c for c in top_n.schema.names
        if c.startswith("buildingblock") and c.endswith("_chemical_id")
    ])

    reconstructed_smiles = []
    for row in top_n.to_pylist():
        smiles_parts = []
        for col in bb_chemical_cols:
            bb_val = row.get(col)
            if bb_val is not None:
                smi = id_to_smile.get(int(bb_val), f"BB:{bb_val}")
                # TODO: RDKit cleanup for top N — sanitize, remove ions, standardize
                # mol = Chem.MolFromSmiles(smi)
                #if mol:
                #    largest_fragment = rdMolStandardize.LargestFragmentChooser().choose(mol)
                #    smi = Chem.MolToSmiles(largest_fragment)
                smiles_parts.append(smi)
        reconstructed_smiles.append(" + ".join(smiles_parts))

    top_n = top_n.append_column("smiles", pa.array(reconstructed_smiles))

    # --- Print ---
    print(f"\n--- Top {n} Disynthons sorted by {sort_by} (Min Occurrences: {min_occurrences}) ---")
    header = f"{'pos_id':<8} | {'origin':<22} | {'pbind':<8} | {'enrich':<8} | {'nhits':<6} | {'ntotal':<7} | {'SMILES'}"
    print(header)
    print("-" * len(header))
    for row in top_n.to_pylist():
        print(
            f"{str(row.get('positional_id', 'N/A')):<8} | "
            f"{str(row.get('origin', 'unknown')):<22} | "
            f"{row.get('pbind', 0.0):<8.4f} | "
            f"{row.get('enrichment', 0.0):<8.4f} | "
            f"{row.get('nhits', 0):<6} | "
            f"{row.get('ntotal', 0):<7} | "
            f"{row.get('smiles', '')}"
        )

    return top_n