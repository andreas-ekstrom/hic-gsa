from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

fig_width_pt = 246.0  # Get this from LaTeX using \showthe\columnwidth
#fig_width_pt = 512.0  # Get this from LaTeX using \showthe\columnwidth
# Get fontsizes from LaTeX with \showthe\f@size

fsize=10 # font size
fsizeSmall=0.9*fsize # 0.833*fsize corresponds to pythons 'small', 0.9 to LaTeX's small
fsizexSmall=0.8*fsize # 0.8 corresponds to LaTeX's \footnotesize
msize=3  # marker size

inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",        # optional, matches LaTeX default font
    "mathtext.default": "regular", # optional
})


rc('axes',labelsize=fsizeSmall)
rc('xtick',labelsize=fsizeSmall)
rc('ytick',labelsize=fsizeSmall)
rc('legend', fontsize = fsizeSmall)
rc('savefig', dpi=300)

col1 = (86/255, 112/255, 173/255, 1.0)
col2 = (207/255, 138/255, 89/255, 1.0)
col3 = (107/255, 165/255, 108/255, 1.0)
col4 = (180/255, 88/255, 85/255, 1.0)

def figure_article(columns=1,rows=1):
    """Creates a pyplot figure with specific matplotlib.rc settings defined in this module.

    ARGUMENT(S):
        columns (integer)
        rows (integer)
        optional arguments (both defaults to 1)
        specifies whether the figure is one- (columns=1) or two-column (columns=2) wide,
        and the number of rows (allowed values = 1,2,3)

    RESULT(S):
        fig_size = [width,height] in inches
    """


    _fig_width  = fig_width*columns                 # widths in inches
    _fig_height = (fig_width)*rows*golden_mean      # height in inches

    fig_size = [_fig_width,_fig_height]

    rc('figure', figsize=fig_size)

    return fig_size

if __name__ == '__main__':
    print('This module contains global variables and settings for publication-ready figures')
