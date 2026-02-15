#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 10:55:48 2020

@author: matthew


To do:
    - update colorbar ticks in results explorer
    - check results explorere works with even number of sources.  
    - colour of labels (blakc, orange, purple).  
"""

import sys
import pickle
from pathlib import Path
import copy
import numpy as np
import matplotlib.pyplot as plt
import pdb


import licsatmo
from licsatmo.licsatmo import LiCSAtmo_correction

# LiCSAtmo requires LiCSAlert as a dependency 
licsalert_dir = str(Path("/home/matthew/university_work/03_automatic_detection_algorithm/06_LiCSAlert/00_LiCSAlert_GitHub").resolve())
if licsalert_dir not in sys.path:
    sys.path.append(licsalert_dir)
import licsalert


#%% Things to set

np.random.seed(41)



#%% sICA at Campi Flegrei


# licsalert_settings = {"figure_type"         : 'both',                             # either 'window' or 'png' (to save as pngs), or 'both'
#                       "downsample_run"      : 0.3,                                     # data can be downsampled to speed things up
#                       "downsample_plot"     : 0.5,                               # and a 2nd time for fast plotting.  Note this is applied to the restuls of the first downsampling, so is compound
#                       }
                      

                     

# icasar_settings = {"sica_tica"              : 'sica',
#                    "n_pca_comp_start"       : 6,                                                  
#                    "n_pca_comp_stop"        : 7,                                                  
#                    "bootstrapping_param"    : (200, 0),                              # (number of runs with bootstrapping, number of runs without bootstrapping)                    "hdbscan_param" : (35, 10),                        # (min_cluster_size, min_samples)
#                     "tsne_param"             : (30, 12),                                       # (perplexity, early_exaggeration)
#                     "ica_param"              : (1e-2, 150),                                     # (tolerance, max iterations)
#                     "hdbscan_param"          : (100,10),                                    # (min_cluster_size, min_samples) Discussed in more detail in Mcinnes et al. (2017). min_cluster_size sets the smallest collection of points that can be considered a cluster. min_samples sets how conservative the clustering is. With larger values, more points will be considered noise. 
#                     "ifgs_format"            : 'cum',                                  # can be 'all', 'inc' (incremental - short temporal baselines), or 'cum' (cumulative - relative to first acquisition)
#                     "load_fastICA_results"   : True}

# licsbas_settings = {"filtered"               : False,
#                     "date_start"            : None,
#                     "date_end"              : None,
#                     'mask_type'             : 'licsbas',                        # "dem" or "licsbas"
#                     'crop_pixels'           : None}


# LiCSAtmo_correction(
#     outdir = Path("./"), 
#     location = "campi_flegrei_022D",
#     automatic_selection = True,                                            
#     licsbas_dir = Path("./example_data/022D_04826_121209_campi_flegrei"),
#     licsalert_settings = licsalert_settings, 
#     icasar_settings = icasar_settings,
#     licsbas_settings = licsbas_settings,
#     xy_list = [(51,26)],
#     )



#%% tICA at Vesuvius

licsalert_settings = {"figure_type"         : 'both',                             # either 'window' or 'png' (to save as pngs), or 'both'
                      "downsample_run"      : 0.3,                                     # data can be downsampled to speed things up
                      "downsample_plot"     : 0.5,                               # and a 2nd time for fast plotting.  Note this is applied to the restuls of the first downsampling, so is compound
                      }
                      


icasar_settings = {"sica_tica"              : 'tica',
                   "n_pca_comp_start"       : 6,                                                  
                   "n_pca_comp_stop"        : 7,                                                  
                   "bootstrapping_param"    : (200, 0),                              # (number of runs with bootstrapping, number of runs without bootstrapping)                    "hdbscan_param" : (35, 10),                        # (min_cluster_size, min_samples)
                    "tsne_param"             : (30, 12),                                       # (perplexity, early_exaggeration)
                    "ica_param"              : (1e-2, 150),                                     # (tolerance, max iterations)
                    "hdbscan_param"          : (32,10),                                    # (min_cluster_size, min_samples) Discussed in more detail in Mcinnes et al. (2017). min_cluster_size sets the smallest collection of points that can be considered a cluster. min_samples sets how conservative the clustering is. With larger values, more points will be considered noise. 
                    "ifgs_format"            : 'cum',                                  # can be 'all', 'inc' (incremental - short temporal baselines), or 'cum' (cumulative - relative to first acquisition)
                    "load_fastICA_results"   : True}

licsbas_settings = {"filtered"               : False,
                    "date_start"            : None,
                    "date_end"              : None,
                    'mask_type'             : 'licsbas',                        # "dem" or "licsbas"
                    'crop_pixels'           : None}


LiCSAtmo_correction(
    outdir = Path("./"), 
    location = "vesuvius_022D_tica",
    automatic_selection = False,
    licsbas_dir = Path("./example_data/022D_04826_121209_vesuvius"),
    licsalert_settings = licsalert_settings, 
    icasar_settings = icasar_settings,
    licsbas_settings = licsbas_settings,
    xy_list = [(36, 22)],
    )




#%% 

