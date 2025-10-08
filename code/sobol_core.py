from SALib.analyze import sobol
from SALib.analyze import hdmr
from SALib.analyze import dgsm
from SALib.analyze import pawn
from SALib.analyze import delta
import numpy as np
import fig_preamble as pre
import matplotlib.pyplot as plt
import math as math

def sensitivity_analysis(problem, Y_values, print_correlations=False):

    problem.set_results(np.array(Y_values))
    problem.analyze_sobol(num_resamples=100, calc_second_order=True, print_to_console=True)  

    Si = problem.analysis
    
    S1_mean, S1_ci = Si['S1'], Si['S1_conf']
    ST_mean, ST_ci = Si['ST'], Si['ST_conf']
   
    return np.array(S1_mean), np.array(S1_ci), np.array(ST_mean), np.array(ST_ci), problem

def sensitivity_analysis_plot_second_order_heatmap(
    problem,
    Y_values,                    # REQUIRED: passed to your wrapper
    xlist_label=None,            # LaTeX ok; defaults to problem.problem['names']
    thresh=0.01,                 # hide |S2| below this
    triangle="upper",            # "upper" or "lower"
    annotate=False,              # write S2 numbers in cells
):
    """
    Plot a Sobol' second-order (S2) heatmap using sensitivity_analysis wrapper.
    Returns (fig, ax).
    """
    _, _, _, _, analyzed = sensitivity_analysis(problem, Y_values, print_correlations=False)

    # Try to get DataFrames; fall back to analysis dict
    df_first = df_total = df_second = None
    try:
        dfs = analyzed.to_df()
        if isinstance(dfs, tuple):
            for df in dfs:
                if df is None:
                    continue
                cols = set(df.columns)
                if "S2" in cols:
                    df_second = df
                elif "S1" in cols:
                    df_first = df
                elif "ST" in cols:
                    df_total = df
    except Exception:
        pass

    names = xlist_label if xlist_label is not None else analyzed.problem["names"]
    d = len(names)

    # Build symmetric S2 matrix
    S2 = np.zeros((d, d), dtype=float)
    if df_second is not None and len(df_second):
        for _, r in df_second.iterrows():
            i, j = int(r["i"]), int(r["j"])
            val = float(r["S2"])
            S2[i, j] = val
            S2[j, i] = val
    else:
        S2_arr = analyzed.analysis.get("S2", None)
        if S2_arr is None:
            raise RuntimeError("No S2 available. Ensure sampling & analysis used calc_second_order=True.")
        S2 = np.array(S2_arr, copy=True)

    # Mask diagonal and one triangle
    if triangle not in ("upper", "lower"):
        triangle = "upper"
    mask = np.triu(np.ones_like(S2, dtype=bool), k=1) if triangle == "upper" \
           else np.tril(np.ones_like(S2, dtype=bool), k=-1)

    # Apply threshold
    S2_plot = np.where(mask & (np.abs(S2) >= float(thresh)), S2, np.nan)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(S2_plot, origin="lower", aspect="auto")
    ax.set_xticks(range(d)); ax.set_xticklabels(names, rotation=90)
    ax.set_yticks(range(d)); ax.set_yticklabels(names)
    ax.set_title("Sobol’ second-order indices (S2)")
    fig.colorbar(im, ax=ax, label="S2")

    if annotate:
        for i in range(d):
            for j in range(d):
                if np.isfinite(S2_plot[i, j]):
                    ax.text(j, i, f"{S2_plot[i, j]:.3f}", ha="center", va="center")

    fig.tight_layout()
    return fig


def sensitivity_analysis_plot_multi(
    problem,
    Y_values_list,           # list of length Ny, each an array-like of model outputs
    xlist_label,             # list of parameter labels (LaTeX ok)
    hist_ranges=None,        # list of (lo, hi) ranges per Y; or None for auto
    y_labels=None,           # list of legend / histogram xlabels (e.g., [r'$E(0^+)$', ...])
    width=None,              # bar width; if None, chosen based on Ny
    capsize=2,               # error bar cap size
):
    """
    Generalized Sobol bar+hist plot for Ny outputs (1..6).
    Requires: `sensitivity_analysis(problem,Y_values)` and `pre` colors.
    """
    Ny = len(Y_values_list)
    assert 1 <= Ny <= 6, "Ny must be between 1 and 6"
    D = len(xlist_label)

    # --- run sensitivity for each Y ---
    results = []
    for k in range(Ny):
        Si_mean, Si_ci, St_mean, St_ci, _ = sensitivity_analysis(problem, Y_values_list[k])
        results.append({
            "Si": 100*np.asarray(Si_mean),
            "Si_ci": 100*np.asarray(Si_ci),
            "St": 100*np.asarray(St_mean),
            "St_ci": 100*np.asarray(St_ci),
        })

    # --- colors: use pre.col1..pre.col6 if available, else fallback to mpl cycle ---
    try:
        pre_colors = [getattr(pre, f"col{i}") for i in range(1, 7) if hasattr(pre, f"col{i}")]
    except NameError:
        pre_colors = []
    if len(pre_colors) < Ny:
        mpl_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        # extend as needed
        while len(pre_colors) < Ny:
            pre_colors += mpl_cycle
            if not mpl_cycle:  # absolute fallback
                pre_colors += ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        pre_colors = pre_colors[:Ny]

    # --- layout: bars on the left, histograms on the right (2 rows) ---
    hist_cols = int(np.ceil(Ny / 2))
    bar_cols = 3  # keep bars wide
    total_cols = bar_cols + (hist_cols if hist_cols > 0 else 0)

    # make hist ranges list-friendly
    if hist_ranges is None:
        hist_ranges = [None] * Ny
    elif len(hist_ranges) != Ny:
        # pad/truncate gracefully
        hist_ranges = (list(hist_ranges) + [None]*Ny)[:Ny]

    if y_labels is None:
        y_labels = [f"Y{k+1}" for k in range(Ny)]

    # sensible default width: keep groups compact as Ny grows
    if width is None:
        width = min(0.25, 0.9 / max(2, Ny + 1))

    ind = np.arange(D)
    fig_size = pre.figure_article(rows=1.54, columns=3) if hasattr(pre, "figure_article") else (10, 5)
    fig = plt.figure(figsize=[fig_size[0], fig_size[1]])
    grid = plt.GridSpec(2, total_cols, width_ratios=[1]*bar_cols + [0.6]*hist_cols)

    # --- histograms (right side), 2 rows by hist_cols columns ---
    for k, Yk in enumerate(Y_values_list):
        row = k % 2         # 0,1,0,1,...
        col = bar_cols + (k // 2)  # 0,0,1,1,2,2 -> shifted by bar_cols
        axh = plt.subplot(grid[row, col:col+1])
        rng = hist_ranges[k]
        if rng is None:
            axh.hist(Yk, bins=200, color=pre_colors[k])
        else:
            axh.hist(Yk, bins=200, range=(rng[0], rng[1]), color=pre_colors[k])
        axh.set_xlabel(y_labels[k])
        axh.locator_params(axis='y', nbins=4)
        axh.ticklabel_format(axis='y', style='sci', scilimits=(0, 1))

    # --- grouped bars (left side) ---
    ax = plt.subplot(grid[0:, 0:bar_cols])

    # center-aligned offsets: [-1.5w, -0.5w, +0.5w, +1.5w] etc.
    offsets = [(k - 0.5*(Ny-1)) * width for k in range(Ny)]

    for k in range(Ny):
        Si = results[k]["Si"];   Si_ci = results[k]["Si_ci"]
        St = results[k]["St"];   St_ci = results[k]["St_ci"]
        x = ind + offsets[k]

        # draw ST (white, underlay) then S1 (colored, overlay)
        ax.bar(x, St, width, yerr=St_ci, color='white', edgecolor='black', capsize=capsize)
        ax.bar(x, Si, width, yerr=Si_ci, color=pre_colors[k], edgecolor='black', capsize=capsize,
               label=f'{y_labels[k]} $S_i$ (main)')

    ax.set_ylabel('Sensitivity (percent)')
    ax.set_xticks(ind)
    ax.set_xticklabels(xlist_label, rotation=60)
    ax.legend()

    return fig


def moving_average(X,Y,nbins):

    bins = np.linspace(X.min(),X.max(), nbins)
    delta = bins[1]-bins[0]
    idx  = np.digitize(X,bins)
    ymean = [np.mean(Y[idx==k]) for k in range(nbins)]
    xmean = bins-delta/2
    
    return xmean, ymean


def main_effect_plot(problem, X_values, Y_values, labels, Nr, Nc):

    # get the number of main effects to analyze
    N = X_values.shape[1]
    fig_size = pre.figure_article(rows=Nr-1,columns=Nc-1)
    fig, axs = plt.subplots(Nr, Nc, figsize = fig_size)
    for this_row in range(0,Nr):
        for this_col in range(0,Nc):
            idx = this_row*Nc+this_col
            print(this_row,this_col,idx)
            if idx>=N:
                break
            axs[this_row,this_col].scatter(X_values[:,idx],Y_values, s=5, color='black', alpha=0.05)
            axs[this_row,this_col].set_xlabel(labels[idx])

            #compute average, i.e., E[Y|X_i]
            xx,yy = moving_average(X_values[:,idx],Y_values,nbins=20)
            axs[this_row,this_col].plot(xx,yy,color='red',lw=3,alpha=0.8)
            axs[this_row,this_col].axhline(10/3,color='green',lw=3,alpha=0.8,ls='--')

    return fig

    
def sensitivity_analysis_radial_plot(problem, Y_values):

    Si = sobol.analyze(problem, np.array(Y_values), print_to_console=True)
    
    result = {} #create dictionary to store new
    result['S1']={k : float(v) for k, v in zip(problem["names"], Si["S1"])}
    result['S1_conf']={k : float(v) for k, v in zip(problem["names"], Si["S1_conf"])}
    result['S2'] = S2_to_dict(Si['S2'], problem)
    result['S2_conf'] = S2_to_dict(Si['S2_conf'], problem)
    result['ST']={k : float(v) for k, v in zip(problem["names"], Si["ST"])}
    result['ST_conf']={k : float(v) for k, v in zip(problem["names"], Si["ST_conf"])}

    fig = drawgraphs(result)

    return fig
