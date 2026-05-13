import DEL_iver as deliv

# =============================================================================
# USER INPUTS
# Change these to match your experiment before running.
# =============================================================================

output  = "./Results"               # Directory where all outputs will be written
input   = "./data/example.csv"      # Path to your DEL screening CSV file

bb_cols = ["bb1_smiles", "bb2_smiles", "bb3_smiles"]   # Column names for building block SMILES
label   = "binds"                   # Column name for the binary hit label (1 = hit, 0 = non-hit)
metric  = "pbind"                   # Metric to rank by: "pbind" (hit rate) or "enrichment"

# =============================================================================
# STEP 1 — Load dataset
# Converts the CSV to Parquet on first run (cached for subsequent runs).
# Cache location is either system default or location given to output_dir
# =============================================================================
ddr = deliv.DataReader.from_csv(input, building_blocks=bb_cols, output_dir=output, label=label)

# =============================================================================
# STEP 2 — Enumerate building blocks
# Assigns each unique building block SMILES a chemical ID and positional ID,
# then computes Cantor-pair IDs for every disynthon combination.
# Results are written to cache and reused by all downstream steps.
# =============================================================================
deliv.enumerate_building_blocks(ddr)

# =============================================================================
# STEP 3 — Compute enrichment
# For every building block position and disynthon pair, calculates:
#   - pbind    : raw hit rate (nhits / ntotal)
#   - enrichment : hit rate relative to the per-position baseline
# min_occurrences filters out building blocks seen fewer than N times.
# =============================================================================
deliv.compute_enrichment(ddr, min_occurrences=0)

# Print a summary table of enrichment statistics across all positions.
# Output file can be specified instead of printing
deliv.data_set_statistics(ddr, print_output=False, write_output="dataset_statistics.csv")

# =============================================================================
# STEP 4 — Identify top-performing building blocks and disynthons
# find_best_bb / find_best_disynthon read the cached enrichment results and
# return the top-n ranked by the chosen metric.
#
# exclude: 1-based position indices (bb) or tuples of indices (disynthon)
#          to omit from ranking — useful for excluding scaffold positions.
# =============================================================================
top_bb = deliv.find_best_bb(
    ddr, 10,
    min_occurrences=30,
    sort_by=metric,
    exclude=[1],            # Exclude BB1 (typically the scaffold/linker position)
)

top_disynthons = deliv.find_best_disynthon(
    ddr, 10,
    min_occurrences=1,
    sort_by=metric,
    exclude=[(1, 2), (1, 3)],   # Exclude disynthon pairs that include BB1
)

# =============================================================================
# STEP 5 — Visualize structures
# draw_bb / draw_disynthons render 2D molecular grids using RDKit.
# Saves SVG for publication-quality figures; omit save paths to display only. (jupyter notebook)
# =============================================================================
deliv.draw_bb(top_bb, ddr, metric="pbind", save_png_path="bb_structures.png")

deliv.draw_disynthons(top_disynthons, ddr, metric=metric, save_png_path="disynthon_structures.png")

# =============================================================================
# STEP 6 — Visualize enrichment landscape
# plot_disynthons: 3D scatter of disynthon pbind/enrichment by positional ID.
# plot_bb:         2-panel scatter of BB pbind by positional ID.
# Both save figure for publication but if you omit saving it gets displayed (jupyter)
# =============================================================================
deliv.plot_disynthons(ddr, elev=15, azim=35, min_occurrences=30, output_path="disynthons.png")
deliv.plot_bb(ddr, output_path="bbs.png", exclude_bb1=True)
