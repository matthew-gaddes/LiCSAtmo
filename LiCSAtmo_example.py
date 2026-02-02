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


# LiCSAtmo requires LiCSAlert as a dependency 


licsalert_dir = str(Path("/home/matthew/university_work/03_automatic_detection_algorithm/06_LiCSAlert/00_LiCSAlert_GitHub").resolve())
if licsalert_dir not in sys.path:
    sys.path.append(licsalert_dir)

import licsalert

#%% Things to set

np.random.seed(42)


#%% Begin


def LiCSAtmo_correction(
        outdir, location,  
        automatic_selection = True,                                            
        licsbas_dir = None,
        licsbas_jasmin_dir = None,
        data_as_arg = None,
        alignsar_dc = None,
        licsalert_settings = None, 
        icasar_settings = None, 
        licsbas_settings = None,
        licsalert_pkg_dir = None,
        ):                   
    """T
    Inputs:
        outdir | pathlib Path | parent out directory.  
        region | string or None | When processing large numbers of volcanoes, it can be ueful to group them into regions.  
                                  Outputs would then be outdir / region / volcano.  
        volcano | string | Name of volcano, but only used to name out directories so could be anything.  

        
        Three options to pass data to the function.  Choose one of the three methods
            licsbas_dir  | pathlib Path | If you have used  LiCSBAS, simply provide the directory of the LiCSBAS outputs.                          
            licsbas_jasmin_dir | pathlib Path | If using a LiCSBAS time series that was automatically created on Jasmin for volcano monitoring, 
                                                simply give the directory of the .json files.  
            data_as_arg | dict of dicts | displacement_r2: dict_keys(['dem', 'mask', 'incremental', 'lons', 'lats']) 
                                          tbaseline_info: dict_keys(['acq_dates', 'ifg_dates', 'baselines', 'baselines_cumulative'])
                                          If there are 327 acq dates (as per the example), there are 327 ifg_dates (as these are the short
                                          temporal baseline ifgs joining the acquisitions, )
        
        licsalert_settings | dict | (['baseline_end', 'figure_intermediate', 
                                      'figure_type', 'downsample_run', 
                                      'downsample_plot', 'residual_type'])
        icasar_settings | dict | (['n_comp', 'bootstrapping_param', 
                                   'tsne_param', 'ica_param', 'hdbscan_param', 
                                   'sica_tica', 'ifgs_format']) 

    Returns:

        
    History:
        2020/06/29 | MEG | Written as a script
 
     """
    import sys
    import os
    import pickle
    import shutil
    import copy
    from datetime import datetime
    import numpy as np
    import numpy.ma as ma
    
    
    def create_licsalert_outdir(outdir):
        """This directory may already exist (if processing was done but 
        errors occured).  Make it again if so.  
        """
        if not os.path.exists(outdir):                                      
            os.mkdir(outdir)                                            
        else:
            print(f"The folder {outdir} appears to exist already.  This is usually "
                  f"due to the date not having all the required LiCSAlert products, "
                  f"and LiCSAlert is now trying to fill this date again.  ")
            shutil.rmtree(outdir)
            os.mkdir(outdir)
    


    
    # if a directory for the package has been provided, assume not on path.
    # possibly only needed for Jasmin?  
    if licsalert_pkg_dir is not None:
        sys.path.append(str(licsalert_pkg_dir))                            
        
    from licsatmo.data_importing import import_insar_data
    from licsatmo.licsatmo import licsatmo_preprocessing
    from licsatmo.aux import r2_to_r3
    from licsatmo.plotting import plot_ifgs_corrected_residual
    from licsatmo.plotting import plot_pixel_history_two_r3

    from licsalert.monitoring_functions import read_config_file
    from licsalert.monitoring_functions import manual_mask_wrapper
    from licsalert.data_exporting import save_licsalert_aux_data
    from licsalert.licsalert import load_or_create_ICASAR_results
    from licsalert.licsalert import licsalert_date_obj
    
    from licsalert.temporal import calculate_all_temporal_info
    from licsalert.aux import Tee# , col_to_ma    
    from licsalert.plotting import LiCSAlert_aux_figures
    from licsalert.licsalert import bss_components_inversion_per_epoch
    
    
    
    #from licsatmo.licsatmo import construct_baseline_ts
    from licsalert.licsalert import construct_baseline_ts

    
    
    # 1: Log all outputs to a file for that location:
    location_dir = outdir / location
    location_dir.mkdir(parents=True, exist_ok=True)         
    # remove as now output to location_dir
    del outdir                          
    
    # append to the single txt file for that volcano                                 
    f_run_log = open(location_dir / "LiCSAtmo_history.txt", 'a')                                                                          
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f_run_log)

    print(f"\n\n\n\n\nLiCSAtmo is being run for {location} at "
          f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")                              


    # if using jasmin data, import the settings for that volcano
    if licsbas_jasmin_dir is not None:
        outputs = read_config_file(
            location_dir /  "LiCSAlert_settings.txt"
            )                                          
        (licsalert_settings, icasar_settings, licsbas_settings) = outputs
        del outputs

    # 2: Open the data.  Note that displacement_r3['cum_ma'] has the time-varying
    # nan mask applied, but whatever mask is selected (in licsbas_settings)
    # is only stored as displacement_r3['mask'], and not applied.  
    displacement_r3, tbaseline_info = import_insar_data(
        location, 
        location_dir,
        location,
        licsalert_settings,
        icasar_settings, 
        licsbas_settings,
        licsbas_jasmin_dir,
        licsbas_dir,
        alignsar_dc,
        data_as_arg,
        )

    print("Applying the mask that is consistent through all times to the"
          "time series.  ")
    
    displacement_r3['cum_ma'].mask = np.repeat(
        displacement_r3['mask'][np.newaxis,],
        repeats = displacement_r3['cum_ma'].shape[0],
        axis = 0
        )
    
    
    # debug
    # from licsalert.debugging import interactive_ts_viewer
    # interactive_ts_viewer(displacement_r3['cum_ma'])

    
    # 3: Possibly draw a mask manually (and apply it to displacement_r3)
    if 'draw_manual_mask' in licsbas_settings.keys():
        displacement_r3 = manual_mask_wrapper(
            location_dir,
            licsbas_settings['draw_manual_mask'],
            displacement_r3
            )        
    

    # Downsample the data in space once for use (to increase speed)
    # and once for plotting (to make very small images)
    # also mean centre in time and space, making 'cum_ma' a mean centered array
    # (a custom LiCSAlert object)
    displacement_r3 = licsatmo_preprocessing(
        displacement_r3,
        tbaseline_info,
        icasar_settings['sica_tica'],                                                    
        licsalert_settings['downsample_run'], 
        licsalert_settings['downsample_plot']
        )
    
    
        # add the temporal baselines (in days) for single master ifgs. relative
     # to the first acquisition
    tbaseline_info=calculate_all_temporal_info(tbaseline_info)
     
    
     # set to end of time series to avoid significant crop in time.  
    baseline_end = licsalert_date_obj(
                tbaseline_info['acq_dates'][-1],
                tbaseline_info['acq_dates'],
                )
    
    
    
    # conver the data to a form that can be used with ICASAR
    displacement_r2_ica = {
        'lons_mg' : displacement_r3['lons_mg'] ,
        'lats_mg' : displacement_r3['lats_mg'] ,
        'dem' : displacement_r3['dem'] ,
        #'mask' : displacement_r3['mask'] ,
        }
    
    from licsatmo.aux import r3_to_r2
    r2_data = r3_to_r2(displacement_r3['cum_ma'].original)
    ifgs_cum = r2_data['ifgs']
    mask = r2_data['mask']
    displacement_r2_ica['mask'] = mask
    displacement_r2_ica['cumulative'] = ifgs_cum
    displacement_r2_ica['incremental'] = np.diff(
        ifgs_cum, 
        axis=0,
        )
    
    
    from copy import deepcopy
    tbaseline_info_ica = deepcopy(tbaseline_info)
    
    
    
    # # determine the time series to be used for ICA.  Note that this is
    # # no longer all epochs, and is instead a subset that compromises
    # # temporal resolution and the number of pixels retained.  
    # # pass full time series and end of baseline info so that function 
    # # can crop to only baseline
    # # mean centereing is redone here.  Not stricly needed in space, 
    # # but needed in time when epochs are dropped
    # displacement_r2_ica, tbaseline_info_ica = construct_baseline_ts(
    #      icasar_settings['sica_tica'],
    #      displacement_r3,
    #      tbaseline_info,
    #      baseline_end,
    #      location_dir,
    #      licsalert_settings['figure_type'],
    #      interactive=False,                # useful to set to True to debug
    # )

    
    # check for the unusual case that there are fewer pixels than pca_comps 
    # requested
    if icasar_settings['sica_tica'] == 'sica':
        if (displacement_r2_ica['incremental'].shape[1] < 
            icasar_settings['n_pca_comp_start']):
            raise Exception(
                "There are fewer pixels "
                "{displacement_r2['incremental'].shape[1]} than principal"
                "components.  The suggest there are very, very few "
                "coherent pixels.  "
                )
         
    # either load ICA from previous run, or compute it.  
    # note that displacement_r2_ica contains mixtures_mc, which are the 
    # input ifgs either mean centered in time or space, depending on 
    # how the sica_tica flag is set
    # note that this now only accepts non-mean-centered data    
    # determine if we need to run ICASAR
    try:
        ICASAR_files = [f.name for f in os.scandir(location_dir / 'ICASAR_results')] 
    except:
        ICASAR_files = []
    if 'ICASAR_results.pkl' in ICASAR_files:
        run_ICASAR = False
        print("\nExisting ICASAR results were found, so these will "
              "not be recalculated.  Delete ICASAR_results.pkl from "
              "the ICASAR_results directory to avoid this. \n ")
    else:
        run_ICASAR = True    
    
       
    
    # # update naming of arg here
    # if icasar_settings['figures'] == 'both':
    #     icasar_settings['figures'] = "png+window"
        
    
    # Main ICA algorithm, note that returns incremental time courses
    outputs = load_or_create_ICASAR_results(
        run_ICASAR,
        displacement_r2_ica,
        tbaseline_info_ica,
        baseline_end,
        location_dir / "ICASAR_results", 
        icasar_settings
    )        
    (icasar_sources, mask_icasar, ics_labels, tcs) = outputs; del outputs
    
    
    # # debug: plot tcs 
    # for n, tc in enumerate(tcs.T):
    #     f, axes = plt.subplots(1,2)
    #     axes[0].plot(tc)
    #     axes[1].plot(np.cumsum(tc))
    #     for ax in axes:
    #         ax.grid(True)
    
    # convert from incremental time courses to cumulative ones.  
    ctcs = np.vstack((
        np.zeros((1, tcs.shape[1])),
        np.cumsum(tcs, axis = 0),
        ))
    
    
    # get hte topo correlated source(s)
    if automatic_selection:
        # if automtic, there is only one chose.  
        src_ns = int(np.where(ics_labels['labels'][:, 1] == 1)[0][0])
    else:
        # Ask user for a list of sources to discard
        user_input = input(
            "Enter a list of integers (comma or space separated) "
            "of the source number(s) to discard (the first source is 0):\n"
            )
        
        # Parse into a list of ints
        src_ns = [int(x) for x in user_input.replace(",", " ").split()]
        
    print("Sources seclected to be discarded:", src_ns)
        
        

            
    # also plot the ICS and the DEM (done once)
    LiCSAlert_aux_figures(
        location_dir,
        icasar_sources,
        displacement_r3['dem'], 
        mask_icasar
    ) 
        
    # also plot some info (e.g. DEM, input data), once.  
    save_licsalert_aux_data(
        location_dir,
        displacement_r3,
        tbaseline_info,
        icasar_settings['sica_tica']
        )


    if icasar_settings['sica_tica'] == 'sica':
        
        # fit each epoch using the ICs.  Note that this function will
        # handle masks that change at each epoch by finding the set of pixels
        # that are valid in both the epoch and the ICs.  
        tcs_c, d_hat, d_resid=bss_components_inversion_per_epoch(
                icasar_sources,
                mask_icasar,
                displacement_r3['cum_ma'].mean_centered.space,
                cumulative=False,
                )
        
        
        # discard the topo correlated sources in space (S) and time (A)
        # (we discard from highest to lowest so after array changes in size
        # indexing doesn't change.  )
        A = tcs_c
        S = icasar_sources
        for src_n in sorted(src_ns)[::-1]:
            A_corrected = np.delete(A, src_n, axis=1)
            S_corrected = np.delete(S, src_n, axis=0)
        
        # reconstruct the time series 
        X_r2_corrected = A_corrected @ S_corrected        
        
        # remove the mean centering in space
        means=np.repeat(
            displacement_r3['cum_ma'].means.space[:, np.newaxis],
            X_r2_corrected.shape[1],
            axis=1,
            )
        X_r2_corrected += means

        # convert 
        X_r3 = r2_to_r3(X_r2_corrected, mask_icasar)


    
        plot_ifgs_corrected_residual(
            displacement_r3['cum_ma'].original,
            X_r3,
            plot_n=-1,
            cmap="viridis",
            titles=("Original", "Reconstruciton (without APS)", "Residual"),
            robust=False,
            robust_pct=(2, 98),
            show_axes=False,
            )
        
        fig, axes = plot_pixel_history_two_r3(
            displacement_r3["cum_ma"].original,
            X_r3,
            x=51, y=26,
            acq_dates=tbaseline_info["acq_dates"],
            labels=("Observed", "Model"),
            window=11,
        )
        

        
    elif icasar_settings['sica_tica'] == 'tica':
        pass
    
        from licsalert.licsalert import bss_components_inversion
        from licsatmo.aux import r3_to_r2
        
        # get the r3 data
        ifgs_for_inv_r3 = displacement_r3['cum_ma'].mean_centered.time
        
        # returns a dict with 'ifgs' and 'mask'
        ifgs_for_inv_r2 = r3_to_r2(ifgs_for_inv_r3)
        
        pdb.set_trace()
        
        m, d_hat, d_resid = bss_components_inversion(
                ctcs.T,
                ifgs_for_inv_r2['ifgs'].T,
                cumulative=False,
                mask = None,
                )
        
        
        # inversion with cumulative time courses (which are S here) to get A (which are images)
        
        # discard
        
        # remake
        
        # remove mean centering
        
        # plot.  
        
    
    

        pdb.set_trace()

    
    sys.stdout = original                                                                                                                                      # return stdout to be normal.  
    f_run_log.close()                                                                                                                                          # and close the log file.  


    return []

#%%


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
#     )



#%%

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
    )




#%% 

sys.exit()

