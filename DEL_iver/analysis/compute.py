from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from DEL_iver.utils.cache import CacheManager, CacheNames
from tqdm import tqdm


def _load_tables(ddr, building_blocks,split_col=None): #!april7 correct
    bb_dict_path = ddr.cache.get_path(
        CacheNames.BB_DICTIONARIES,
        filename=f"{CacheNames.BB_DICTIONARIES.value}.{ddr.source_file.stem}.parquet"
    )
    if not ddr.cache.is_cached(bb_dict_path):
        raise RuntimeError("BB dictionaries not found. Run generate_bb_dictionaries(ddr) first.")

    columns = building_blocks + [ddr.label]

    if split_col is not None:
        columns.append(split_col)

    bb_table = pq.read_table(bb_dict_path)
    source_table = pq.read_table(ddr.source_file, columns=columns)

    for col in bb_table.schema.names:
        source_table = source_table.append_column(col, bb_table[col])

    return source_table, bb_table


def _get_global_totals(source_table, label_col): #!april7 correct
    total_hits = pc.sum(pc.equal(source_table[label_col], 1)).as_py()
    total_nonhits = pc.sum(pc.equal(source_table[label_col], 0)).as_py()
    return total_hits, total_nonhits


def _count_hits_and_total(table, group_col, label_col,extra_col=None):  #TODO: MIGHT BE BUGGED
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


def _aggregate_bb_counts_across_positions(source_table, bb_id_cols, label_col,ignore_position=False): #TODO: MIGHT BE BUGGED

    chunks = []
    for bb_id_col in bb_id_cols:
        chunk = source_table.select([bb_id_col, label_col]).rename_columns(["id", label_col])
        chunk = chunk.append_column("origin", pa.array([bb_id_col] * len(chunk)))
        chunks.append(chunk)
    stacked = pa.concat_tables(chunks)


    # --- Global counts (shared IDs pooled together) ---
    if ignore_position:
        total = stacked.group_by("id").aggregate([(label_col, "count")]).rename_columns(["id", "ntotal"])
        filtered = stacked.filter(pc.equal(stacked[label_col], 1)) #!only hits 
        hits = filtered.group_by("id").aggregate([(label_col, "count")]).rename_columns(["id", "nhits"]) #!
        counts = total.join(hits, "id", join_type="left outer")
        pooled_array = pa.array(["pooled"] * len(counts))
        counts = counts.append_column("origin", pooled_array)
        #!^ all positions get merged into a single column, and then grouped by id disregards origin
    else:
        total = stacked.group_by(["id","origin"]).aggregate([(label_col, "count")]).rename_columns(["id", "origin", "ntotal"])
        filtered = stacked.filter(pc.equal(stacked[label_col], 1)) #!only hits 
        hits = filtered.group_by(["id","origin"]).aggregate([(label_col, "count")]).rename_columns(["id", "origin", "nhits"]) 
        counts = total.join(hits, keys=["id", "origin"], join_type="left outer")


    return counts



def _apply_enrichment(stats, total_hits, total_nonhits, method, min_occurrences): #!works correctly

    #Extract the nhits column
    nhits_column = stats["nhits"]

    # Identify null values in nhits
    nhits_is_null_mask = pc.is_null(nhits_column)

    # Replace nulls with 0
    nhits_no_nulls = pc.if_else(nhits_is_null_mask, 0, nhits_column)

    #Cast nhits to integer now that there are no null
    nhits_int = nhits_no_nulls.cast(pa.int32())

    #Extract ntotal column and cast to integer
    ntotal_int = stats["ntotal"].cast(pa.int32())

    # Replace the old nhits column with cleaned integer version
    stats = stats.set_column(stats.schema.get_field_index("nhits"), "nhits", nhits_int)

    # Replace the old ntotal column with integer version
    stats = stats.set_column(stats.schema.get_field_index("ntotal"), "ntotal", ntotal_int)

    # Filter the table first
    if min_occurrences > 0:
        stats = stats.filter(pc.greater_equal(stats["ntotal"], min_occurrences))

    #name n_total column and nhits column for further use
    nhits = stats["nhits"]
    ntotal = stats["ntotal"]

    nnonhits_column = pc.subtract(ntotal, nhits)

    stats = stats.append_column("nnonhits", nnonhits_column)
    nnonhits=stats["nnonhits"]

    pbind = pc.divide(nhits.cast(pa.float64()), ntotal.cast(pa.float64()))
    stats = stats.append_column("pbind", pbind)

    if method.lower() == 'laplace':
        f_hits = pc.divide(pc.add(nhits.cast(pa.float64()), 1.0), float(total_hits))
        f_nonhits = pc.divide(pc.add(nnonhits.cast(pa.float64()), 1.0), float(total_nonhits))
    elif method.lower() == 'epsilon':
        eps = 1e-6
        f_hits = pc.add(pc.divide(nhits.cast(pa.float64()), float(total_hits)), eps)
        f_nonhits = pc.add(pc.divide(nnonhits.cast(pa.float64()), float(total_nonhits)), eps)
    else:
        raise ValueError("Method must be 'laplace' or 'epsilon'")

    stats = stats.append_column("f_hits", f_hits)
    stats = stats.append_column("f_nonhits", f_nonhits)
    stats = stats.append_column("enrichment", pc.divide(f_hits, f_nonhits))

    return stats



def _count_hits_and_total_disynthon(table, dis_col, bb_id_cols, label_col):
    group_cols = [dis_col] + bb_id_cols
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
    return total.join(hits, group_cols, join_type="left outer")


def _compute_bb_enrichment(source_table, bb_id_cols, label_col, total_hits, total_nonhits, method, min_occurrences,ignore_position:bool):
    

    counts = _aggregate_bb_counts_across_positions(source_table, bb_id_cols, label_col,ignore_position) #!ALL GOOD UP TO HERE IS CORRECT


    stats = _apply_enrichment(counts, total_hits, total_nonhits, method, min_occurrences) #!ALL GOOD UP TO HERE IS CORRECT
    #stats = _attach_bb_smiles(stats, id_to_smile) #!ALL GOOD UP TO HERE IS CORRECT
    return stats.append_column("type", pa.array(["building_block"] * len(stats))) #!ALL GOOD UP TO HERE IS CORRECT


def _compute_disynthon_enrichment(source_table, dis_col:str,relevant_bb_id_cols, label_col, total_hits, total_nonhits, method, min_occurrences):

    renamed = source_table.rename_columns({dis_col: "id"} if dis_col != "id" else {})


    counts = _count_hits_and_total_disynthon(renamed, "id", relevant_bb_id_cols,label_col)

    stats = _apply_enrichment(counts, total_hits, total_nonhits, method, min_occurrences)
    #stats = _attach_disynthon_smiles(stats, source_table, dis_col, bb_smi_cols)
    return stats.append_column("type", pa.array([dis_col] * len(stats)))


def _write_output(ddr, table):
    output_path = ddr.cache.get_path(
        CacheNames.COMPUTE,
        filename=f"{CacheNames.COMPUTE.value}.{ddr.source_file.stem}.parquet"
    )
    pq.write_table(table, output_path)
    print(f"Written to {output_path}")

#TODO: NEED TO ADD FLAG, SO THAT THERE CAN BE TWO BEHAVIORS, ONE WHERE EACH SMILE IS TREATED UNIQULY PER POSITION, ONE THAT IS POSITION AGNOSTIC(CURRENT)
def compute_pbind_and_enrichment(ddr, method='laplace', min_occurrences=0,ignore_position: bool =False,draw: bool = False):
    building_blocks = ddr.building_blocks 

    source_table, bb_table = _load_tables(ddr, building_blocks)
    disynthon_cols = ddr.disynthons #!ALL GOOD UP TO HERE IS CORRECT dont like strings but ce la vi
    bb_id_cols = [f"{bb}_id" for bb in building_blocks]


    total_hits, total_nonhits = _get_global_totals(source_table, ddr.label) #!ALL GOOD UP TO HERE IS CORRECT

    bb_stats = _compute_bb_enrichment( #!ALL GOOD UP TO HERE IS CORRECT
        source_table, 
        bb_id_cols, 
        ddr.label, 
        total_hits, 
        total_nonhits, 
        method, 
        min_occurrences,
        ignore_position
    )

    disynthon_stats = {}
    for dis_col in tqdm(ddr.disynthons, desc="Computing disynthon pbind"):
        combo_indices = [int(i) - 1 for i in dis_col.replace("disynthon_", "").replace("_id", "").split("_")] #TODO: replace with cleaner logic
        relevant_bb_id_cols = [bb_id_cols[i] for i in combo_indices]
        result = _compute_disynthon_enrichment(
            source_table,
            dis_col,
            relevant_bb_id_cols,
            ddr.label,
            total_hits,
            total_nonhits,
            method,
            min_occurrences
        )
        disynthon_stats[dis_col]=result

    final_table = pa.concat_tables([bb_stats] + list(disynthon_stats.values()), promote_options="default")


    _write_output(ddr, final_table)
    print("it worked")
    return bb_stats, disynthon_stats




import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

def find_best_bb(ddr, n, min_occurrences=0, sort_by="pbind", exclude: list =None):
    """
    Find the top N building blocks by pbind or enrichment score, including their SMILES.

    Args:
        ddr: DDR object with cache and source_file attributes
        n: Number of top building blocks to return
        min_occurrences: Minimum occurrences filter (reserved for future filtering)
        sort_by: Column to sort by — either "pbind" or "enrichment"
        exclude: List of building block origin names to exclude, e.g.
                 ["buildingblock1_smiles_id"] to only see BB2 and BB3

    Returns:
        top_n (pa.Table): Top N building blocks sorted by chosen metric descending,
                          now including a 'smiles' column.
    """
    if sort_by not in ("pbind", "enrichment"):
        raise ValueError(f"sort_by must be 'pbind' or 'enrichment', got '{sort_by}'")

    # --- Load enrichment table (COMPUTE cache) ---
    enrichment_path = ddr.cache.get_path(
        CacheNames.COMPUTE,
        filename=f"{CacheNames.COMPUTE.value}.{ddr.source_file.stem}.parquet"
    )
    enrichment = pq.read_table(enrichment_path)

    # --- Load id-to-smiles table (BB_DICTIONARIES cache) ---
    id_to_smile_path = ddr.cache.get_path(
        CacheNames.BB_DICTIONARIES,
        filename=f"{CacheNames.BB_DICTIONARIES.value}.{ddr.source_file.stem}_id_to_smiles.parquet"
    )
    id_to_smile_table = pq.read_table(id_to_smile_path)

    id_to_smile = {
        int(row["id"]): row["smiles"]
        for row in id_to_smile_table.to_pylist()
    }

    # --- Filter to only building block rows (not disynthons) ---
    if "type" in enrichment.schema.names:
        bb_stats = enrichment.filter(pc.equal(enrichment["type"], "building_block"))
    else:
        bb_stats = enrichment

    exclude = [exclude] if isinstance(exclude, str) else (exclude or [])
    building_blocks = ddr.building_blocks 
    bb_id_cols = [f"{bb}_id" for bb in building_blocks]

    # map int to ddr.building_blocks 
    mapped_exclude = []
    for item in exclude:
        if isinstance(item, int):
            # Translate '1' to building_blocks[0], '2' to building_blocks[1], etc.
            if 1 <= item <= len(bb_id_cols):
                mapped_exclude.append(bb_id_cols[item - 1])
            else:
                raise IndexError(f"Exclude index {item} is out of bounds for building_blocks of size {len(building_blocks)}.")

    # --- Apply exclude filter by origin, then deduplicate by id ---
    if mapped_exclude:
        if "origin" in bb_stats.schema.names:
            # Check if we are in 'pooled' mode
            is_pooled = bb_stats["origin"].nbytes > 0 and bb_stats["origin"][0].as_py() == "pooled"
            
            if is_pooled:
                print(f"Warning: Cannot exclude {exclude} because stats were pooled across all positions.")
            else:
                # Normal filtering logic
                keep_mask = pc.invert(pc.is_in(bb_stats["origin"], value_set=pa.array(mapped_exclude)))
                bb_stats = bb_stats.filter(keep_mask)

    # --- Validate sort column exists ---
    if sort_by not in bb_stats.schema.names:
        raise ValueError(
            f"Column '{sort_by}' not found in table. "
            f"Available columns: {bb_stats.schema.names}"
        )

    # --- Sort by chosen metric descending and take top N ---
    bb_stats_sorted = bb_stats.sort_by([(sort_by, "descending")])
    top_n = bb_stats_sorted.slice(0, n)

    # --- Inject the SMILES column into the PyArrow Table ---
    top_ids = top_n["id"].to_pylist()
    smiles_list = [id_to_smile.get(i, "SMILES not found") for i in top_ids]
    
    # Append the new column directly to the table
    top_n = top_n.append_column("smiles", pa.array(smiles_list))

    # --- Print top N with SMILES ---
    for row in top_n.to_pylist():
        bb_id = int(row["id"])
        origin = row.get("origin", "Pooled") 
        smile = row.get("smiles")  # Now pulling straight from the row!
        score = row.get(sort_by, 0.0)
        
        print(f"ID: {bb_id} | Origin: {origin} | {sort_by}: {score:.4f} | SMILES: {smile}")

    return top_n

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

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

    # --- Filter to only disynthon rows ---
    if "type" in enrichment.schema.names:
        dis_stats = enrichment.filter(pc.invert(pc.equal(enrichment["type"], "building_block")))
    else:
        dis_stats = enrichment

    # --- Apply min_occurrences filter ---
    if min_occurrences > 0:
        dis_stats = dis_stats.filter(pc.greater_equal(dis_stats["ntotal"], min_occurrences))

    # --- Apply exclude filter ---
    if mapped_exclude and dis_stats.num_rows > 0:
        available_types = pc.unique(dis_stats["type"]).to_pylist()
        actual_excludes = [e for e in mapped_exclude if e in available_types]
        if actual_excludes:
            keep_mask = pc.invert(pc.is_in(dis_stats["type"], value_set=pa.array(actual_excludes)))
            dis_stats = dis_stats.filter(keep_mask)

    if dis_stats.num_rows == 0:
        print("No disynthon data found after filtering.")
        return dis_stats

    # --- Sort and Slice ---
    dis_stats_sorted = dis_stats.sort_by([(sort_by, "descending")])
    top_n = dis_stats_sorted.slice(0, n)

    # --- Construct SMILES Column ---
    # Identify BB ID columns present in the schema
    bb_cols = sorted([c for c in top_n.schema.names if c.startswith("buildingblock") and c.endswith("_id")])
    
    reconstructed_smiles = []
    for row in top_n.to_pylist():
        smiles_parts = []
        for col in bb_cols:
            bb_val = row.get(col)
            if bb_val is not None:
                smiles_parts.append(id_to_smile.get(int(bb_val), f"BB:{bb_val}"))
        
        reconstructed_smiles.append(" + ".join(smiles_parts))
    
    # Append the column to the table
    top_n = top_n.append_column("smiles", pa.array(reconstructed_smiles))

    # --- Print top N ---
    print(f"\n--- Top {n} Disynthons by {sort_by} (Min Occurrences: {min_occurrences}) ---")
    for row in top_n.to_pylist():
        print(f"ID: {row['id']:<6} | Type: {row['type']:<18} | "
              f"{sort_by}: {row[sort_by]:.4f} | ntotal: {row['ntotal']:<5} | "
              f"SMILES: {row['smiles']}")

    return top_n