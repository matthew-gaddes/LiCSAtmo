#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 15:06:39 2026

@author: matthew
"""

import pdb



def LiCSAtmo_correction(
        outdir, location,  
        xy_list,
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
    from licsatmo.aux import r2_to_r3, r3_to_r2
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
    from licsalert.licsalert import bss_components_inversion
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
        A_corrected = tcs_c
        S_corrected = icasar_sources

        for src_n in sorted(src_ns, reverse=True):
            A_corrected = np.delete(A_corrected, src_n, axis=1)
            S_corrected = np.delete(S_corrected, src_n, axis=0)
        
        # reconstruct the time series 
        X_r2_corrected = A_corrected @ S_corrected        
        
        # remove the mean centering in space
        means=np.repeat(
            displacement_r3['cum_ma'].means.space[:, np.newaxis],
            X_r2_corrected.shape[1],
            axis=1,
            )
        X_r2_corrected += means
        
    elif icasar_settings['sica_tica'] == 'tica':
        # get the r3 data (mean centered in time)
        ifgs_for_inv_r3 = displacement_r3['cum_ma'].mean_centered.time
        
        # returns a dict with 'ifgs' and 'mask'
        ifgs_for_inv_r2 = r3_to_r2(ifgs_for_inv_r3)
        
        
        # mean centre the time courses
        ctcs_mc = ctcs - np.mean(ctcs, axis=0)
        
        # invert to fit time series with cumulative time courses
        # m is n_pixels x n_ics
        m, d_hat, d_resid = bss_components_inversion(
                ctcs_mc.T,
                ifgs_for_inv_r2['ifgs'].T,
                cumulative=False,
                mask = None,
                )
        
        # discard the topo correlated sources in space (A) and time (S)
        # (we discard from highest to lowest so after array changes in size
        # indexing doesn't change.  )
        # A is n_pixels x n_ics and S is n_ics x _times
        A_corrected = m
        S_corrected = ctcs_mc.T

        for src_n in sorted(src_ns, reverse=True):
            A_corrected = np.delete(A_corrected, src_n, axis=1)
            S_corrected = np.delete(S_corrected, src_n, axis=0)


        # reconstruct the time series (still mean centered)
        X_r2_corrected = A_corrected @ S_corrected  
        
        # remove the mean centering in space
        means_r1 = ma.compressed(displacement_r3['cum_ma'].means.time)
        means=np.repeat(
            means_r1[:, np.newaxis],
            X_r2_corrected.shape[1],
            axis=1,
            )
        X_r2_corrected += means

        # convert back to spatial orientation for consistencty in plotting
        # i.e. n_times x n_pixels
        X_r2_corrected = X_r2_corrected.T


    # convert back to rank 3 
    X_r3_corrected = r2_to_r3(X_r2_corrected, mask_icasar)


    # figure outputs
    # A: Total time series, original, corrected, difference
    plot_ifgs_corrected_residual(
        displacement_r3['cum_ma'].original,
        X_r3_corrected,
        plot_n=-1,
        cmap="viridis",
        titles=("Original", "Reconstruciton (without APS)", "Residual"),
        robust=False,
        robust_pct=(2, 98),
        show_axes=False,
        )
    
    # correction for a pixels of interest.     
    for x, y in xy_list:
        fig, axes = plot_pixel_history_two_r3(
            displacement_r3["cum_ma"].original,
            X_r3_corrected,
            x=x,
            y=y,
            acq_dates=tbaseline_info["acq_dates"],
            labels=("Observed", "Model"),
            window=11,
        )
            

    sys.stdout = original                                                                                                                                      # return stdout to be normal.  
    f_run_log.close()                                                                                                                                          # and close the log file.  


    return []

def licsatmo_preprocessing(
        displacement_r3,
        tbaseline_info, 
        sica_tica,
        downsample_run=1.0,
        downsample_plot=0.5,
        verbose=True,
        order = 1,
        anti_aliasing=False
    ):
    """A function to downsample the data at two scales (one for general working [ie to speed things up], and one 
    for faster plotting.  )  Also, data are mean centered, which is required for ICASAR and LiCSAlert.  
    Note that the downsamples are applied consecutively, so are compound (e.g. if both are 0.5, 
    the plotted data will be at 0.25 the resolution of the original data).  
    
    Inputs:
        displacement_r2 | dict | input data stored in a dict as row vectors with a mask    
                                 Also lons and lats as rank 2 arrays.  E N U are components of the look vector for East North Up for each pixel.  
                                 dict_keys(['dem', 'mask', 'incremental', 'lons', 'lats', 'E', 'N', 'U'])
        tbaseline_info | dict |  dict_keys(['acq_dates', 'ifg_dates', 'baselines', 'baselines_cumulative'])
        sica_tica       | string | if sica, spatial ICA, if tica, temporal ica
        downsample_run | float | in range [0 1], and used to downsample the "incremental" data
        downsample_plot | float | in range [0 1] and used to downsample the data again for the "incremental_downsample" data
        verbose | boolean | if True, informatin returned to terminal
        
        order : int, default 1 | Interpolation order passed to `rescale` 
                                    (1 = bilinear, 0 = nearest, etc.).
        anti_aliasing : bool, default False Forwarded to `rescale`.  Set True 
                                    for images, False for scientific rasters.
        
    Outputs:
        displacement_r2 | dict | input data stored in a dict as row vectors with a mask
                                 updated so that "incremental" is downsampled (and its mask), 
                                 and a new key is created, called "incremental_downsampled" 
                                 that is downsamled further for fast plotting                                 
                                 
                                 'dem' - the DEM, downsampled.  
                                 'mask' - mask, downsampled.  
                                 'incremental' - daisy chain of short temporal baseline ifgs, downsampled but not mean centered.  
                                 'lons' - longitude for each pixel, downsampled.  
                                 'lats' - latitude for each pixel, downsmapled. 
                                 'E' - east component of look vector, downsmapled.  
                                 'N' - north component of look vector, downsmapled.
                                 'U' - up componennt of look vector, downsampled.  
                                 'incremental_downsampled' - as above, but downsampled again (so downsampled_run * downsample_plot)
                                 'mask_downsampled' - mask for double downsamled data.  
                                 
                                 'incremental_mc_space' - as for incremental, but mean cenetered in space.  
                                 'means_space' - to undo mean centering in space.  
                                 'incremental_mc_time' - as for incremental, but mean centered in time. 
                                 'means_time' - to undo mean centering in time.  
                                 
                                 'mixtures_mc' - mean centered in space or time, depending on how sica_tica was set.  
                                 'means' - to undo above mean centering.  
                                 
                                 
    History:
        2020/01/13 | MEG | Written
        2020/12/15 | MEG | Update to also downsample the lons and lats in the ICASAR geocoding information.  
        2021_04_14 | MEG | Update so handle rank2 arrays of lons and lats properly.  
        2021_05_05 | MEG | Add check that lats are always the right way up, and fix bug in lons.  
        2021_10_13 | MEG | Add function to also downsample ENU grids.  
        2021_11_15 | MEG | Add option to control mean centering, and warning that it is happening.  
        2023_10_26 | MEG | use ifg_timeseries class to handle mean centering more carefully.  
        
        2025_06_22 | MEG | WIP - had to remove mean centerering stuff 
    """
    import numpy as np
    import numpy.ma as ma
    # from licsalert.downsample_ifgs import downsample_ifgs
    from licsalert.aux import col_to_ma
    from licsalert.data_importing import ifg_timeseries
    
    from skimage.transform import rescale

    # class MeanCenteredArray:
    #     def __init__(self, masked_array):
    #         if not isinstance(masked_array, np.ma.MaskedArray):
    #             raise TypeError("Input must be a masked array")
    #         # interal/private attriubute
    #         self._arr = masked_array
    #         self.mean_centered = _MeanCenteredAccessor(self._arr)
    
    #     @property
    #     def original(self):
    #         return self._arr
    
    
    # class _MeanCenteredAccessor:
    #     def __init__(self, arr):
    #         self._original = arr
    
    #         # Precompute mean-centered versions
    #         # Mean-centre in space (each image mean = 0)
    #         image_means = arr.mean(axis=(1, 2))  # shape (time,)
    #         self._space_centered = arr - image_means[:, np.newaxis, np.newaxis]
    
    #         # Mean-centre in time (each pixel's time series mean = 0)
    #         pixel_means = arr.mean(axis=0)  # shape (y, x)
    #         self._time_centered = arr - pixel_means[np.newaxis, :, :]
    
    #     @property
    #     def space(self):
    #         """Return image-wise mean-centred array (mean over y,x = 0 for each time slice)."""
    #         return self._space_centered
    
    #     @property
    #     def time(self):
    #         """Return time-wise mean-centred array (mean over time = 0 for each pixel)."""
    #         return self._time_centered

    import numpy as np
    
    class MeanCenteredArray:
        def __init__(self, masked_array):
            if not isinstance(masked_array, np.ma.MaskedArray):
                raise TypeError("Input must be a masked array")
            
            self._arr = masked_array
    
            # Precompute means
            self._image_means = self._arr.mean(axis=(1, 2))  # shape (time,)
            self._pixel_means = self._arr.mean(axis=0)       # shape (y, x)
    
            # Accessors
            self.mean_centered = _MeanCenteredAccessor(self._arr, self._image_means, self._pixel_means)
            self.means = _MeanAccessor(self._image_means, self._pixel_means)
    
        @property
        def original(self):
            return self._arr
    
    
    class _MeanCenteredAccessor:
        def __init__(self, arr, image_means, pixel_means):
            self._space_centered = arr - image_means[:, np.newaxis, np.newaxis]
            self._time_centered = arr - pixel_means[np.newaxis, :, :]
    
        @property
        def space(self):
            """Mean-centre in space: each 2D slice has zero mean."""
            return self._space_centered
    
        @property
        def time(self):
            """Mean-centre in time: each pixel's time series has zero mean."""
            return self._time_centered
    
    
    class _MeanAccessor:
        def __init__(self, image_means, pixel_means):
            self._image_means = image_means
            self._pixel_means = pixel_means
    
        @property
        def space(self):
            """Return 1D array of means over space (y,x) for each time step."""
            return self._image_means
    
        @property
        def time(self):
            """Return 2D array of mean time series per pixel (shape: y, x)."""
            return self._pixel_means

    
    
    def rescale_masked(arr_ma, scale, *, order=1, anti_aliasing=False,
                       **kw):
        """
        Down-/up-scale a 3-D MaskedArray with skimage.rescale
        while preserving the mask.
    
        Parameters
        ----------
        arr_ma : np.ma.MaskedArray   (t, y, x)
            Data cube; mask may be nomask.
        scale  : float | tuple
            Scale tuple to feed straight into skimage.rescale
            (for your use-case: (1.0, f, f)).
        order, anti_aliasing, **kw
            Forwarded to skimage.transform.rescale.  `preserve_range`
            is forced to True so numeric range/dtype survive.
    
        Returns
        -------
        np.ma.MaskedArray
            Same rank as input, spatially rescaled, with a rebuilt mask.
        """
        
        import numpy.ma as ma
    
        # --- 1. replace masked values with NaN so they don’t bias interpolation
        data_in = arr_ma.filled(np.nan)
    
        data_out = rescale(
            data_in,
            scale=scale,
            order=order,
            anti_aliasing=anti_aliasing,
            preserve_range=True,      # keep original numeric scale
            **kw
        )
    
        # --- 2. new mask: every NaN generated above becomes masked
        mask_out = np.isnan(data_out)
    
        # --- 3. pack back into a MaskedArray and restore original dtype
        return ma.MaskedArray(
            data_out.astype(arr_ma.dtype, copy=False),
            mask=mask_out
        )
        

    # n_pixs_start = displacement_r2["incremental"].shape[1]               
    # shape_start = displacement_r2["mask"].shape                          
    
    # 1: Downsample the ifgs for use in all following functions.  
    # if we're not actually downsampling, skip for speed
    if downsample_run != 1.0:
        
        # downsample the cumulative masked array
        print(
            f"The cumulative time series was size "
            f"{displacement_r3['cum_ma'].shape} ",end=''
            )
        
        # note that the rescale in time (first dimension) is held at 1.  
        displacement_r3['cum_ma'] = rescale_masked(
            displacement_r3['cum_ma'],
            scale=(1.0, downsample_run, downsample_run),
            order=order,
            anti_aliasing=anti_aliasing,
        )
        
        print(
            f"and has been downsampled in space to "
            f"{displacement_r3['cum_ma'].shape} "
            )
        
        # remake lons and lats at new resolution
        # check if we have lon lat data as not alway strictly necessary.  
        if ('lons_mg' in displacement_r3) and ('lats_mg' in displacement_r3):

            lons = np.linspace(
                displacement_r3['lons_mg'][0,0], 
                displacement_r3['lons_mg'][-1,-1], 
                displacement_r3['cum_ma'].shape[2]
                )
            displacement_r3['lons_mg'] = np.repeat(
                lons[np.newaxis, :], 
                displacement_r3['cum_ma'].shape[1],
                axis = 0
                )                       
            
            lats = np.linspace(
                displacement_r3['lats_mg'][0,0], 
                displacement_r3['lats_mg'][-1,-1], 
                displacement_r3['cum_ma'].shape[1]
                )
            displacement_r3['lats_mg'] = np.repeat(
                lats[:, np.newaxis], 
                displacement_r3['cum_ma'].shape[2],
                axis = 1
                )                       
            
            # poor quality check to ensure that lats aren't upside down            
            if displacement_r3['lats_mg'][0,0] < displacement_r3['lats_mg'][-1,0]:                                        
                displacement_r3['lats_mg'] = np.flipud(displacement_r3['lats_mg'])                                        
            
        # also downsample other simple data if it's included:
        for product in ['dem', 'E', 'N', 'U']: 
            if product in displacement_r3.keys():
                displacement_r3[product] = rescale(
                    displacement_r3[product], 
                    downsample_run, 
                    anti_aliasing = anti_aliasing
                )                               
            
    # 2: Downsample further for plotting.  
    # note that the rescale in time (first dimension) is held at 1.  
    displacement_r3['cum_ma_downsampled'] = rescale_masked(
        displacement_r3['cum_ma'],
        scale=(1.0, downsample_run, downsample_run),
        order=order,
        anti_aliasing=anti_aliasing,
    )
    
    
    # 3: also compute the incremental displacements
    displacement_r3['inc_ma'] = ma.diff(
        displacement_r3['cum_ma'],
        axis=0
        )
    
    displacement_r3['inc_ma_downsampled'] = ma.diff(
        displacement_r3['cum_ma_downsampled'],
        axis=0
        )
    

    # debug    
    # pdb.set_trace()
    # f, ax = plt.subplots()
    # ax.matshow(displacement_r3['cum_ma'].mask[1])
    
    # pdb.set_trace()
    
    # deal with mean centering (which could be in space or time)
    displacement_r3['cum_ma'] = MeanCenteredArray(displacement_r3['cum_ma'])
    displacement_r3['inc_ma'] = MeanCenteredArray(displacement_r3['inc_ma'])
   
    return displacement_r3

#############################################################################

# def construct_baseline_ts(
#         sica_tica, 
#         displacement_r3, 
#         tbaseline_info,
#         volcano_dir,
#         figures='png',
#         interactive=False
#         ):
#     """
#     A function to prepare the baseline time series for ICA.  First step
#     is dropping some epochs that cause many pixels to be lost, 
    
#     Note that data mustn't be mean centered.  
    
#     Inputs:
#         sica_tica | str | either 'sica' or 'tica'
#         displacement_r3 | dict | time series info as rank 3
#         tbaseline_info | dict | temporal info associated with time series. 
#         volcano_dir | Path | outdir for current volcano.  
#         figures | str | png, window, or both. 
#         interactive | boolean | interacitve figure to explore how the trade off
#                                 between number of epochs and number of pixels. 
#                                 (i.e. few epochs means lots of pixels, lots of 
#                                  epochs means few pixels, usually)
                                
#     Returns:
        
#     History:
#         2025_07_?? | MEG | Written.  
    
#     """
#     import numpy as np
    

#     from licsalert.temporal import daisy_chain_from_acquisitions
#     from licsalert.data_importing import ifg_timeseries
    
    

    
#     # 1: select a subset of the epochs to create a compromise between lots of 
#     # pixels and lots of epochs (i.e. drop the epochs that cause lots of 
#     # pixels to be lost when a consistent mask through time is sought)
#     # this returns cumulative displacements that are not mean centered
#     displacement_r2_ica, tbaseline_info_ica = automatic_pixel_epoch_selection(
#         displacement_r3,
#         tbaseline_info,
#         volcano_dir,
#         figures,
#         interactive,
#         )
    
#     # also copy some auxilliary info to the new array
#     for key in ['dem', 'lons_mg', 'lats_mg']:
#         displacement_r2_ica[key] = displacement_r3[key]
    
#     # rename the output to be more readable
#     displacement_r2_ica['cumulative'] = displacement_r2_ica['ifgs']
    
#     del displacement_r2_ica['ifgs']
    
#     # f, ax = plt.subplots()
#     # ax.matshow(displacement_r2_ica['cumulative'])
#     # ax.set_aspect('auto')
    
#     # create the incremental displacments
#     displacement_r2_ica['incremental'] = np.diff(
#         displacement_r2_ica['cumulative'],
#         axis = 0
#         )
    
#     # calculate the dates of the incremental ifgs
#     tbaseline_info_ica['ifg_dates'] = daisy_chain_from_acquisitions(
#         tbaseline_info_ica['acq_dates']
#     )
        

#     return displacement_r2_ica, tbaseline_info_ica
    

# ##############################################################################

    


# def automatic_pixel_epoch_selection(
#         displacement_r3, 
#         tbaseline_info,
#         volcano_dir,
#         figures,
#         interactive=False
#         ):
#     """
#     Given a time series with a time varying mask (i.e. pixels come 
#     in and out of coherene), build a time series with a consistent mask
#     that uses only some of these acquisitions to build a compromise 
#     between temporal resolution and number of pixels
    
    
#     Inputs
    
#     spatial_ICASAR_data = {'ifgs_dc'       : displacement_r2['mixtures_mc'][:(baseline_end.acq_n+1),],                             
#                            'mask'          : displacement_r2['mask'],
#                            'lons'          : displacement_r2['lons'],
#                            'lats'          : displacement_r2['lats'],
                           
                           
#    'ifg_dates_dc'  : tbaseline_info['ifg_dates'][:(baseline_end.acq_n+1)]}                             
    
#     volcano_dir | Path | outdir for figures.  
    
#     """
    
#     import numpy as np
#     import numpy.ma as ma
    
#     from licsalert.pixel_selection import calculate_valid_pixels
#     from licsalert.pixel_selection import calculate_optimal_n_epochs
#     from licsalert.pixel_selection import consistent_pixels_plot
#     from licsalert.pixel_selection import intersect_valid_pixels
#     from licsalert.aux import r3_to_r2


#     # crop the input data in time
#     cum_ma_baseline = displacement_r3['cum_ma'].original
#     acq_dates_baseline = tbaseline_info['acq_dates']
    
#     # determine the number of pixels for each epoch
#     n_pixels, n_pixels_idx, total_pix = calculate_valid_pixels(
#         cum_ma_baseline
#         )
    
#     # and how those change as we add epochs
#     n_pix_epoch = intersect_valid_pixels(
#         cum_ma_baseline,
#         acq_dates_baseline,
#         verbose = False
#         )
    
#     # calculate optimal number of epochs
#     epoch_values, optimal_epoch_n = calculate_optimal_n_epochs(
#         n_pix_epoch,
#         volcano_dir,
#         figures,
#         )
#     # tidy as not needed.  
#     del epoch_values
    
    
#     # Figure, which can be set to interactive for debugging.  
#     consistent_pixels_plot(
#         cum_ma_baseline,
#         acq_dates_baseline,
#         volcano_dir,
#         figures,
#         optimal_epoch_n,
#         interactive,
#         )
        
    
#     # build the time series that uses this number of epochs
#     selected_epochs = sorted(list(n_pixels_idx[:optimal_epoch_n]))    
#     cum_ma_ica = cum_ma_baseline[selected_epochs, ]
#     acq_dates_ica = [acq_dates_baseline[i] for i in selected_epochs]
    
#     # convert to ICA data form
#     # debug            
#     # import matplotlib.pyplot as plt
#     # for im in cum_ma_ica:
#     #     f, ax = plt.subplots(1)
#     #     ax.matshow(im)
    
#     mask_bool = ma.getmaskarray(cum_ma_ica)   # nomask → array(False)
#     # 2. Logical OR along the time axis:  True if masked at least once
#     mask_2d = np.any(mask_bool, axis=0)
    
#     # f, ax = plt.subplots()
#     # ax.matshow(mask_2d)
    
#     # make the mask that's consistent in time.  
#     mask_3d = np.repeat(
#         mask_2d[np.newaxis,],
#         cum_ma_ica.shape[0],
#         axis = 0
#         )
    
#     # and apply to the data.  
#     cum_ma_ica_consistent = cum_ma_ica.copy()
#     cum_ma_ica_consistent.mask = mask_3d
    
#     # debug plot
#     # for im in cum_ma_ica_consistent:
#     #     f, ax = plt.subplots()
#     #     ax.matshow(im)
#     #     plt.pause(0.5)
        
#     # flatten to ICA standard (image is a row vector)            
#     displacement_r2_ica = r3_to_r2(cum_ma_ica_consistent)

#     tbaseline_info_ica = {
#         'acq_dates' : acq_dates_ica
#         }

#     return displacement_r2_ica, tbaseline_info_ica
