import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from DEL_iver.utils.cache import CacheManager, CacheNames

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
import random
import matplotlib.colors as mcolors
from rdkit.Chem.MolStandardize import rdMolStandardize

from IPython.display import display, SVG

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_disynthons(ddr, 
                    mode="pbind", 
                    min_occurrences=3,
                    log_scale=False, 
                    elev=30, 
                    azim=45,
                    font_name='Arial',
                    output_path:str=None,
                    ):
    
    # 1. Setup Plot Parameters
    params = {
        'font.family': 'sans-serif',
        'legend.fontsize': 16, 
        'figure.figsize': (16, 8),
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    }

    # 2. Load and Filter Data
    input_path = ddr.cache.get_path(
        CacheNames.COMPUTE,
        filename=f"{CacheNames.COMPUTE.value}.{ddr.source_file.stem}.parquet"
    )

    #print(f"Reading data from: {input_path}")
    df = pd.read_parquet(input_path)

    if "ntotal" in df.columns:
        df = df[df["ntotal"] >= min_occurrences].copy()

    bb_positional_id_cols = [f"{bb}_positional_id" for bb in ddr.building_blocks]
    dis_cols = [t for t in df['origin'].unique() if str(t).startswith("disynthon_")]
    
    if not dis_cols:
        print("No disynthon types found in the dataset.")
        return

    metrics = []
    if mode in ["pbind", "both"]: metrics.append(("pbind", "Pbind"))
    if mode in ["enrichment", "both"]: metrics.append(("enrichment", "Enrichment"))
    
    n_rows = len(dis_cols)
    n_cols = len(metrics)
    bb_labels = {col: f"BB{i + 1}" for i, col in enumerate(bb_positional_id_cols)}
    
    # 3. Apply Plot Parameters and Draw
    with plt.rc_context(params):
        # Increased base height slightly to accommodate the larger padding
        fig = plt.figure(figsize=(12 * n_cols, 10 * n_rows)) 
        colormap = LinearSegmentedColormap.from_list("magenta_cyan", ["magenta", "cyan"])

        for row_idx, dis_col in enumerate(dis_cols):
            df_dis = df[df["origin"] == dis_col].copy()
            
            if df_dis.empty:
                continue

            try:
                parts = dis_col.replace("disynthon_", "").replace("_id", "").split("_")
                combo_indices = [int(i) - 1 for i in parts]
            except ValueError:
                continue
            
            if len(combo_indices) != 2:
                print(f"Skipping {dis_col}: 3D plotting requires exactly 2 building blocks.")
                continue

            x_col = bb_positional_id_cols[combo_indices[0]]
            y_col = bb_positional_id_cols[combo_indices[1]]
            x_label = bb_labels[x_col]
            y_label = bb_labels[y_col]
            bb_pair_label = f"{x_label}_{y_label}"

            for col_idx, (metric_col, label) in enumerate(metrics):
                if metric_col not in df_dis.columns:
                    continue

                plot_index = (row_idx * n_cols) + col_idx + 1
                ax = fig.add_subplot(n_rows, n_cols, plot_index, projection='3d')
                
                plot_data = df_dis.dropna(subset=[x_col, y_col, metric_col]).copy()
                
                if plot_data.empty:
                    continue

                x_vals, y_vals = plot_data[x_col], plot_data[y_col]

                z_raw = plot_data[metric_col].clip(lower=1e-6)
                z = np.log10(z_raw) if log_scale else plot_data[metric_col]
                z_label = f"Log10({label})" if log_scale else label

                sc = ax.scatter(x_vals, y_vals, z, c=z, cmap=colormap, 
                                s=15, alpha=0.7, edgecolor=(0, 0, 0, 0),
                                linewidth=0.1, zorder=1)

                fig.colorbar(sc, ax=ax, shrink=0.5, aspect=20, label=z_label, pad=0.1)
                
                # --- FIX OVERLAPPING AXES HERE ---
                
                # 1. Slant the tick marks and push them away from the axis line
                ax.tick_params(axis='x', labelrotation=-30, pad=1)
                #ax.tick_params(axis='y', labelrotation=45, pad=1)
                #ax.tick_params(axis='z', pad=15)

                # 2. Push the Axis titles further out from the ticks
                ax.set_title(f"{label} — {bb_pair_label}", fontweight='bold', pad=30)
                ax.set_xlabel(x_label, labelpad=40)
                ax.set_ylabel(y_label, labelpad=40)
                ax.set_zlabel(z_label, labelpad=30)
                
                # ---------------------------------
                
                ax.view_init(elev=elev, azim=azim)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')

        plt.tight_layout()
        plt.show()


def plot_bb(ddr, 
            min_occurrences=3,
            exclude_bb1=False,
            output_path: str = None):
            
    params = {
        'font.family': 'sans-serif',
        'legend.fontsize': 24,
        'figure.figsize': (16, 8),
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24
    }
    
    with plt.rc_context(params):
        # 1. Load Data
        input_path = ddr.cache.get_path(
            CacheNames.COMPUTE,
            filename=f"{CacheNames.COMPUTE.value}.{ddr.source_file.stem}.parquet"
        )
        #print(f"Reading data from: {input_path}")
        df = pd.read_parquet(input_path)

        # 2. Filter Data
        mask = df["type"] == "building_block"
        df = df[mask]
        
        if "ntotal" in df.columns:
            df = df[df["ntotal"] >= min_occurrences].copy()

        if df.empty:
            print("No building_block data left after filtering.")
            return

        # 3. Setup Labels & Mappings
        bb_positional_id_cols = [f"{bb}_positional_id" for bb in ddr.building_blocks]
        bb_labels = {col: f"BB{i + 1}" for i, col in enumerate(bb_positional_id_cols)}
        
        origins = sorted(df['origin'].unique())
        
        if exclude_bb1:
            origins = [origin for origin in origins if bb_labels.get(origin, origin) != "BB1"]

        n_cols = len(origins)
        if n_cols == 0:
            print("No origin categories found (or all were excluded).")
            return

        # 4. Plot Setup
        merge_plots = exclude_bb1 and n_cols > 1
        
        fig_width = max(6, 6 * n_cols)
        fig, axes_grid = plt.subplots(
                    2, n_cols,
                    figsize=(fig_width, 6),
                    gridspec_kw={'height_ratios': [1, 3], 'hspace': 0.15},
                    sharex='col',
                    sharey='row' if merge_plots else False
                )

        # Normalise to always be 2D array shape (2, n_cols)
        if n_cols == 1:
            axes_grid = axes_grid.reshape(2, 1)

        colormap = "cool_r"

        # Find global min and max for ntotal for shared colorbar scaling
        global_vmin = df[df["origin"].isin(origins)]['ntotal'].min() if merge_plots else None
        global_vmax = df[df["origin"].isin(origins)]['ntotal'].max() if merge_plots else None

        last_scatter = None

        # 5. Iterate and Plot
        for i, origin in enumerate(origins):
            ax_top = axes_grid[0, i]  # upper panel: pbind in (0.1, 1.0]
            ax_bot = axes_grid[1, i]  # lower panel: pbind in [0.0, 0.1]

            df_sub = df[df["origin"] == origin]
            if df_sub.empty:
                continue

            x_label = bb_labels.get(origin, origin)

            scatter_kwargs = dict(
                c=df_sub['ntotal'],
                cmap=colormap,
                alpha=0.7,
                linewidth=0,
                s=10,
                vmin=global_vmin,
                vmax=global_vmax
            )

            sc_top = ax_top.scatter(df_sub['positional_id'], df_sub['pbind'], **scatter_kwargs)
            sc_bot = ax_bot.scatter(df_sub['positional_id'], df_sub['pbind'], **scatter_kwargs)
            last_scatter = sc_top

            # Set y-limits defining the break
            ax_top.set_ylim(0.025, 1.0)
            ax_bot.set_ylim(0.0, 0.025)

            ax_top.grid(False)
            ax_bot.grid(False)

            # X-label only on bottom panel
            ax_bot.set_xlabel(f'{x_label} ID', labelpad=20)
            ax_top.set_xlabel('')

            # Y-label only on leftmost column
            if i == 0 or not merge_plots:
                #ax_top.set_ylabel('pBind', labelpad=20)
                ax_bot.set_ylabel('pBind', labelpad=20)

            # Hide the touching spines to create the visual break
            #ax_top.spines['bottom'].set_visible(False)
            #ax_bot.spines['top'].set_visible(False)
            #ax_top.tick_params(axis='x', bottom=False, labelbottom=False)

            # Individual colorbars if NOT merging
        if not merge_plots:
            cbar = fig.colorbar(sc_top, ax=[ax_top, ax_bot],
                                fraction=0.05, pad=0.05)

            # Force scientific formatting for ticks (but no offset text)
            formatter = ticker.ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((0, 0))
            cbar.ax.yaxis.set_major_formatter(formatter)

            # Hide the automatic 10^n offset
            cbar.ax.yaxis.get_offset_text().set_visible(False)

            # Manually include scaling in label
            cbar.set_label(r'Count ($\times 10^{n}$)', fontsize=24, labelpad=10)


        if merge_plots and last_scatter is not None:
                    cbar = fig.colorbar(
                        last_scatter,
                        ax=axes_grid.ravel().tolist(),
                        fraction=0.05,
                        pad=0.05
                    )

                    # 1. Force scientific math
                    formatter = ticker.ScalarFormatter(useMathText=True)
                    formatter.set_scientific(True)
                    formatter.set_powerlimits((0, 0))
                    cbar.ax.yaxis.set_major_formatter(formatter)
                    
                    # 2. Force the figure to render so we can "see" the math
                    fig.canvas.draw() 

                    # 3. Grab the exponent
                    exp = formatter.orderOfMagnitude # Should be 5 for your data
                    
                    # 4. If an exponent was found, manually rescale the tick labels
                    if exp != 0:
                        # Get current tick values (e.g., 200000.0)
                        ticks = cbar.get_ticks()
                        # Create labels divided by the magnitude (e.g., "2.0")
                        # Using :.1f to keep one decimal point like 0.5, 1.0, 1.5, 2.0
                        new_labels = [f'{t/10**exp:.1f}' for t in ticks]
                        ticks = cbar.get_ticks()
                        cbar.set_ticks(ticks)  # lock tick positions
                        cbar.ax.set_yticklabels(new_labels)
                        
                        label_text = f'Count ($10^{{{exp}}}$)'
                    else:
                        label_text = 'Count'

                    # 5. Finalize the side label
                    cbar.set_label(label_text, rotation=270, labelpad=25)
                    
                    # 6. Hide the floating "x 10^5" at the top
                    cbar.ax.yaxis.get_offset_text().set_visible(False)

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {output_path}")
            
        plt.show()

def draw_bb(top_n, ddr, metric="enrichment", mols_per_row=3, remove_ions=True, save_svg_path=None, save_png_path=None):


    mols = []
    legends = []

    origin_to_label = {
        f"{bb}_positional_id": f"BB{i+1}"
        for i, bb in enumerate(ddr.building_blocks)
    }


    largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser() if remove_ions else None

    rows = top_n.to_pylist() if hasattr(top_n, "to_pylist") else top_n


    grouped = {}

    for row in rows:
        chem_id = row.get("chemical_id")
        if chem_id is None:
            continue

        chem_id = int(chem_id)
        grouped.setdefault(chem_id, []).append(row)



    for chem_id, chem_rows in grouped.items():
        smiles = chem_rows[0].get("smiles")

        if not smiles:

            continue

        mol = Chem.MolFromSmiles(smiles)
        if not mol:

            continue

        if remove_ions:
            mol = largest_fragment_chooser.choose(mol)

        mols.append(mol)

        positions = []
        scores = []



        for row in chem_rows:
            origin_val = row.get("origin")
            bb_label = origin_to_label.get(str(origin_val), str(origin_val))
            metric_val = row.get(metric, 0.0)

            positions.append(bb_label)
            scores.append(f"{metric_val:.2f}")




        pos_str = ", ".join(positions)
        score_str = ", ".join(scores)

        legend = f"ID: {chem_id} | Position: {pos_str} | {metric}: {score_str}"


        legends.append(legend)



    if not mols:

        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(400, 400),
        legends=legends,
        useSVG=False
    )

    if save_svg_path:
        svg_string = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(400, 400), legends=legends, useSVG=True)
        with open(save_svg_path, "w") as f:
            f.write(svg_string)
        print(f"Saved SVG with top {len(top_n)} building blocks to: {save_svg_path}")

    if save_png_path:
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(400, 400), legends=legends, useSVG=False)
        img.save(save_png_path)
        print(f"Saved PNG to: {save_png_path}")


    return img



def draw_bb(top_n, ddr, metric="enrichment", mols_per_row=3, remove_ions=True, save_svg_path=None, save_png_path=None):


    mols = []
    legends = []

    origin_to_label = {
        f"{bb}_positional_id": f"BB{i+1}"
        for i, bb in enumerate(ddr.building_blocks)
    }


    largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser() if remove_ions else None

    rows = top_n.to_pylist() if hasattr(top_n, "to_pylist") else top_n


    grouped = {}

    for row in rows:
        chem_id = row.get("chemical_id")
        if chem_id is None:
            continue

        chem_id = int(chem_id)
        grouped.setdefault(chem_id, []).append(row)



    for chem_id, chem_rows in grouped.items():
        smiles = chem_rows[0].get("smiles")

        if not smiles:

            continue

        mol = Chem.MolFromSmiles(smiles)
        if not mol:

            continue

        if remove_ions:
            mol = largest_fragment_chooser.choose(mol)

        mols.append(mol)

        positions = []
        scores = []



        for row in chem_rows:
            origin_val = row.get("origin")
            bb_label = origin_to_label.get(str(origin_val), str(origin_val))
            metric_val = row.get(metric, 0.0)

            positions.append(bb_label)
            scores.append(f"{metric_val:.2f}")




        pos_str = ", ".join(positions)
        score_str = ", ".join(scores)

        legend = f"ID: {chem_id} | Position: {pos_str} | {metric}: {score_str}"


        legends.append(legend)



    if not mols:

        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(400, 400),
        legends=legends,
        useSVG=False
    )

    if save_svg_path:
        svg_string = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(400, 400), legends=legends, useSVG=True)
        with open(save_svg_path, "w") as f:
            f.write(svg_string)
        print(f"Saved SVG to: {save_svg_path}")

    if save_png_path:
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(400, 400), legends=legends, useSVG=False)
        img.save(save_png_path)
        print(f"Saved PNG to: {save_png_path}")


    return img

def draw_disynthons(top_n, ddr, metric="enrichment", mols_per_row=3, remove_ions=True, save_svg_path=None, save_png_path=None):
    mols = []
    legends = []
    origin_to_label = {
        f"{bb}_positional_id": f"BB{i+1}"
        for i, bb in enumerate(ddr.building_blocks)
    }
    largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser() if remove_ions else None

    rows = top_n.to_pylist() if hasattr(top_n, "to_pylist") else top_n

    for row in rows:
        metric_val = row.get(metric, 0.0)
        raw_smiles = row.get("smiles", "")

        fragments = [s.strip() for s in raw_smiles.split("+")]
        clean_fragments = []
        for smi in fragments:
            m = Chem.MolFromSmiles(smi)
            if m and remove_ions:
                m = largest_fragment_chooser.choose(m)
            if m:
                clean_fragments.append(Chem.MolToSmiles(m))

        if not clean_fragments:
            continue

        clean_smiles = ".".join(clean_fragments)

        bb_chemical_cols = sorted([
            f"{bb}_chemical_id"
            for bb in ddr.building_blocks
            if f"{bb}_chemical_id" in row
        ])
        bb_labels = []
        for col in bb_chemical_cols:
            if row[col] is not None:
                chem_id  = int(row[col])
                pos_col  = col.replace("_chemical_id", "_positional_id")
                bb_label = origin_to_label.get(pos_col, col)
                bb_labels.append(f"{bb_label}({chem_id})")

        mol = Chem.MolFromSmiles(clean_smiles)
        if mol:
            mols.append(mol)
            dis_id    = row.get("positional_id", "N/A")
            bb_string = " + ".join(bb_labels)
            legends.append(f"ID: {dis_id} | {metric}: {metric_val:.2f}\n{bb_string}")

    if not mols:
        print("No valid molecules to draw.")
        return

    if save_svg_path:
        svg_string = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(400, 400), legends=legends, useSVG=True)
        with open(save_svg_path, "w") as f:
            f.write(svg_string)
        print(f"Saved SVG to: {save_svg_path}")

    if save_png_path:
        img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(400, 400), legends=legends, useSVG=False)
        img.save(save_png_path)
        print(f"Saved PNG to: {save_png_path}")

    img = Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, subImgSize=(400, 400), legends=legends, useSVG=False)
    return img