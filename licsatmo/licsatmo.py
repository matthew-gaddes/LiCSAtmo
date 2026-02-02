#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 15:06:39 2026

@author: matthew
"""

import pdb

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
