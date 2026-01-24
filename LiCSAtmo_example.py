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



#%% Begin


def LiCSAtmo_correction(
        outdir, location,                                              
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
        

    from licsalert.monitoring_functions import read_config_file
    from licsalert.monitoring_functions import manual_mask_wrapper
    from licsalert.data_importing import import_insar_data
    from licsalert.data_exporting import save_licsalert_aux_data
    from licsalert.licsalert import LiCSAlert_preprocessing, LiCSAlert#, shorten_LiCSAlert_data
    from licsalert.licsalert import write_volcano_status, load_or_create_ICASAR_results
    from licsalert.licsalert import licsalert_date_obj
    from licsalert.licsalert import construct_baseline_ts
    from licsalert.temporal import calculate_all_temporal_info
    from licsalert.aux import Tee, find_nearest_date, col_to_ma
    from licsalert.aux import update_mask
    # from licsalert.downsample_ifgs import downsample_ifgs
    from licsalert.plotting import LiCSAlert_figure, LiCSAlert_epoch_figures
    from licsalert.plotting import LiCSAlert_aux_figures, LiCSAlert_mask_figure
    from licsalert.plotting import create_manual_mask
    
    
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
            location_dir /  "LiCSAlert_settings.txt")                                          
        (licsalert_settings, icasar_settings, licsbas_settings) = outputs
        del outputs

    # 2: Open the data
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
    
    
    # debug
    # from licsalert.debugging import interactive_ts_viewer
    # interactive_ts_viewer(displacement_r3['cum_ma'])
    
    # 3: Possibly draw a mask manually (and apply it to displacement_r2)
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
    displacement_r3 = LiCSAlert_preprocessing(
        displacement_r3,
        tbaseline_info,
        icasar_settings['sica_tica'],                                                    
        licsalert_settings['downsample_run'], 
        licsalert_settings['downsample_plot']
        )
    
    
    
    # save minimal data for example
    # import pickle
    # displacement_r3_example = {}
    # displacement_r3_example['mask']=displacement_r3['mask']
    # displacement_r3_example['dem']=displacement_r3['dem']
    # displacement_r3_example['lons_mg']=displacement_r3['lons_mg']
    # displacement_r3_example['lats_mg']=displacement_r3['lats_mg']
    # displacement_r3_example['cum_ma']=displacement_r3['cum_ma'].original
    # with open('cordon_culle_ts.pkl', 'wb') as f:
    #     pickle.dump(displacement_r3_example, f)
    #     pickle.dump(tbaseline_info, f)
    
    # pdb.set_trace()
    
    # # save all the data
    # import cloudpickle
    # with open('cordon_culle_ts.pkl', 'wb') as f:
    #     cloudpickle.dump(displacement_r3, f)
    #     cloudpickle.dump(tbaseline_info, f)
    
    # add the temporal baselines (in days) for single master ifgs. relative
    # to the first acquisition
    tbaseline_info=calculate_all_temporal_info(tbaseline_info)
    


   
    # determine the time series to be used for ICA.  Note that this is
    # no longer all epochs, and is instead a subset that compromises
    # temporal resolution and the number of pixels retained.  
    # pass full time series and end of baseline info so that function 
    # can crop to only baseline
    # mean centereing is redone here.  Not stricly needed in space, 
    # but needed in time when epochs are dropped
    displacement_r2_ica, tbaseline_info_ica = construct_baseline_ts(
        icasar_settings['sica_tica'],
        displacement_r3,
        tbaseline_info,
        licsalert_settings['baseline_end'],
        location_dir,
        licsalert_settings['figure_type'],
        interactive=False,                # useful to set to True to debug
    )
    
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
    outputs = load_or_create_ICASAR_results(
        True,
        displacement_r2_ica,
        tbaseline_info_ica,
        licsalert_settings['baseline_end'],
        location_dir / "ICASAR_results", 
        icasar_settings
    )        
    (icasar_sources, mask_icasar, ics_labels) = outputs; del outputs


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

    sys.stdout = original                                                                                                                                      # return stdout to be normal.  
    f_run_log.close()                                                                                                                                          # and close the log file.  


#%%


licsalert_settings = {"figure_type"         : 'png',                             # either 'window' or 'png' (to save as pngs), or 'both'
                      "downsample_run"      : 0.3,                                     # data can be downsampled to speed things up
                      "downsample_plot"     : 0.5,                               # and a 2nd time for fast plotting.  Note this is applied to the restuls of the first downsampling, so is compound
                      }
                      

                     

icasar_settings = {"n_pca_comp_start"       : 6,                                                  
                   "n_pca_comp_stop"        : 7,                                                  
                   "bootstrapping_param"    : (200, 0),                              # (number of runs with bootstrapping, number of runs without bootstrapping)                    "hdbscan_param" : (35, 10),                        # (min_cluster_size, min_samples)
                    "tsne_param"             : (30, 12),                                       # (perplexity, early_exaggeration)
                    "ica_param"              : (1e-2, 150),                                     # (tolerance, max iterations)
                    "hdbscan_param"          : (100,10),                                    # (min_cluster_size, min_samples) Discussed in more detail in Mcinnes et al. (2017). min_cluster_size sets the smallest collection of points that can be considered a cluster. min_samples sets how conservative the clustering is. With larger values, more points will be considered noise. 
                    "ifgs_format"            : 'cum',                                  # can be 'all', 'inc' (incremental - short temporal baselines), or 'cum' (cumulative - relative to first acquisition)
                    "load_fastICA_results"   : True}

licsbas_settings = {"filtered"               : False,
                    "date_start"            : None,
                    "date_end"              : None,
                    'mask_type'             : 'licsbas',                        # "dem" or "licsbas"
                    'crop_pixels'           : None}


LiCSAtmo_correction(
    outdir = Path("./"), 
    location = "campi_flegrei_022D",
    licsbas_dir = Path("./example_data/022D_04826_121209_campi_flegrei"),
    licsalert_settings = licsalert_settings, 
    icasar_settings = icasar_settings,
    licsbas_settings = licsbas_settings,
    )




#%% 

sys.exit()

#from licsalert.monitoring_functions import LiCSAlert_monitoring_mode
#from licsalert.licsalert import reconstruct_ts_from_dir

