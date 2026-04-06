from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from DEL_iver.utils.cache import CacheManager, CacheNames
from tqdm import tqdm


def _load_tables(ddr, building_blocks,split_col=None):
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


def _aggregate_bb_counts_across_positions(source_table, bb_id_cols, label_col):  #!works correctly
    # Stack all BB id columns into one long column alongside the label
    chunks=[]

    #stack each buildin block id in a single column called id, including its label value, 
    for bb_id_col in bb_id_cols:
        chunks.append(source_table.select([bb_id_col, label_col]).rename_columns(["id", label_col]))
    stacked = pa.concat_tables(chunks)

    #Filter out non hits
    filtered=stacked.filter(pc.equal(stacked[label_col], 1))

    #group by each unique id value
    grouped_hits = filtered.group_by("id")

    #Count number of rows per group
    aggregated_hits = grouped_hits.aggregate([(label_col, "count")])

    #Rename columns to "id" and "nhits"
    hits = aggregated_hits.rename_columns(["id", "nhits"])

    grouped_total = stacked.group_by("id")

    # Aggregate each group by counting the label_col
    aggregated_total = grouped_total.aggregate([(label_col, "count")])

    # Step 3: Rename the columns to "id" and "ntotal"
    total = aggregated_total.rename_columns(["id", "ntotal"])

    return total.join(hits, "id", join_type="left outer") 



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


def _attach_bb_smiles(stats, id_to_smile): #!ALL GOOD UP TO HERE IS CORRECT
    smiles = pa.array([id_to_smile[i.as_py()] for i in stats["id"]])
    return stats.append_column("smiles", smiles)


def _attach_disynthon_smiles(stats, source_table, dis_col, bb_smi_cols):
    lookup = source_table.select([dis_col] + bb_smi_cols)
    lookup = lookup.group_by(dis_col).aggregate([(col, "min") for col in bb_smi_cols])
    combined_smiles = pc.binary_join_element_wise(
        *[lookup[col] for col in lookup.schema.names if col != dis_col], "."
    )
    lookup = pa.table({"id": lookup[dis_col], "smiles": combined_smiles})
    return stats.join(lookup, "id")

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

def _compute_bb_enrichment(source_table, bb_id_cols, label_col, total_hits, total_nonhits, method, min_occurrences):
    counts = _aggregate_bb_counts_across_positions(source_table, bb_id_cols, label_col) #!ALL GOOD UP TO HERE IS CORRECT
    stats = _apply_enrichment(counts, total_hits, total_nonhits, method, min_occurrences) #!ALL GOOD UP TO HERE IS CORRECT
    #stats = _attach_bb_smiles(stats, id_to_smile) #!ALL GOOD UP TO HERE IS CORRECT
    return stats.append_column("type", pa.array(["building_block"] * len(stats))) #!ALL GOOD UP TO HERE IS CORRECT


def _compute_disynthon_enrichment(source_table, dis_col,relevant_bb_id_cols, label_col, total_hits, total_nonhits, method, min_occurrences):

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


#def compute_pbind_and_enrichment(ddr, method='laplace', min_occurrences=0, include_bb1=True):
#    building_blocks = ddr.building_blocks if include_bb1 else ddr.building_blocks[1:] #TODO:GIVE OPTION TO EXLCUDE WHICHEVER YOUD LIKE

#    source_table, bb_table = _load_tables(ddr, building_blocks)
#    disynthon_cols = [c for c in bb_table.schema.names if c.startswith("disynthon_")] #!ALL GOOD UP TO HERE IS CORRECT dont like strings but ce la vi
#    bb_id_cols = [f"{bb}_id" for bb in building_blocks]


#    total_hits, total_nonhits = _get_global_totals(source_table, ddr.label) #!ALL GOOD UP TO HERE IS CORRECT

#    bb_stats = _compute_bb_enrichment( #!ALL GOOD UP TO HERE IS CORRECT
#        source_table, 
#        bb_id_cols, 
#        ddr.label, 
#        total_hits, 
#        total_nonhits, 
#        method, 
#        min_occurrences
#    )

#    disynthon_stats = []
#    for dis_col in tqdm(disynthon_cols, desc="Computing disynthon pbind"):
#        combo_indices = [int(i) - 1 for i in dis_col.replace("disynthon_", "").replace("_id", "").split("_")]
#        relevant_bb_id_cols = [bb_id_cols[i] for i in combo_indices]
#        result = _compute_disynthon_enrichment(
#            source_table,
#            dis_col,
#            relevant_bb_id_cols,
#            ddr.label,
#            total_hits,
#            total_nonhits,
#            method,
#            min_occurrences
#        )
#        disynthon_stats.append(result)

    #final_table = pa.concat_tables([bb_stats] + disynthon_stats, promote_options="default")





    #TODO: MAKE IT WRITGHT OUT WITH CACHE MANAGER
#    #_write_output(ddr, final_table)
#    print("it worked")
#    return bb_stats, disynthon_stats



def compute_pbind_and_enrichment(ddr, method='laplace', min_occurrences=0, include_bb1=True, split_col=None):
    building_blocks = ddr.building_blocks if include_bb1 else ddr.building_blocks[1:]

    source_table, bb_table = _load_tables(ddr, building_blocks,split_col=split_col)
    disynthon_cols = [c for c in bb_table.schema.names if c.startswith("disynthon_")]
    bb_id_cols = [f"{bb}_id" for bb in building_blocks]
    total_hits, total_nonhits = _get_global_totals(source_table, ddr.label)

    if split_col is None:
        splits = {None: source_table}
    else:
        # Get unique values in split column
        unique_vals = pc.unique(source_table[split_col]).to_pylist()
        splits = {
            val: source_table.filter(pc.equal(source_table[split_col], val))
            for val in unique_vals
        }

    bb_results = {}
    disynthon_results = {}

    for split_val, split_table in splits.items():
        split_total_hits, split_total_nonhits = _get_global_totals(split_table, ddr.label)

        bb_stats = _compute_bb_enrichment(
            split_table, bb_id_cols, ddr.label,
            split_total_hits, split_total_nonhits, method, min_occurrences
        )
        bb_results[split_val] = bb_stats

        dis_stats = {}
        for dis_col in tqdm(disynthon_cols, desc=f"Computing disynthon pbind [{split_val}]"):
            combo_indices = [int(i) - 1 for i in dis_col.replace("disynthon_", "").replace("_id", "").split("_")]
            relevant_bb_id_cols = [bb_id_cols[i] for i in combo_indices]
            dis_stats[dis_col] = _compute_disynthon_enrichment(
                split_table, dis_col, relevant_bb_id_cols, ddr.label,
                split_total_hits, split_total_nonhits, method, min_occurrences
            )
        disynthon_results[split_val] = dis_stats

    # Flatten if no split
    if split_col is None:
        return bb_results[None], disynthon_results[None]

    return bb_results, disynthon_results