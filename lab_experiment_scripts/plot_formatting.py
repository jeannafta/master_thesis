#!/usr/bin/env python
# coding: utf-8

##############################################################################
############################### Font formatting ##############################
##############################################################################

# plt.style.use('seaborn') # uncomment to use seaborn style

params = {# Latex
        "text.usetex": True,
        "font.family": "serif",
        "font.serif" : ["Computer Modern Serif"],
        # Font size (adpated to the font size of my master's thesis which is of 12pts)
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        # Color of ticks and labels
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        }

##############################################################################
############################## Size of figures ###############################
############################################################################## 

# text width of my latex document is 505.89pt.
width = 505.89

# Function taken from: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
# The first parameter of the function has been modified so that the default value 
# is the text width of my latex document.
def set_size(width= width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

