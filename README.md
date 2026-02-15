# LiCSAtmo

LiCSAtmo is a tool to reduce atmospheric noise signals in time series of InSAR data.  It has been developed using time series processed with LiCSBAS from Sentinel-1 interferograms processed by LiCSAR, but can be used with other data.  

The algorithm is based on the assumption that signals such as topographically correlated atmospheric phase screens (APS) are statistically independent in time (and usually space) from deformation, and are non-Gaussian.  We can therefore recover them using temporal independent component analysis (tICA) if there are independent in time, and spatial independent component analysis (sICA) if there are independent in space.  In practise, the huge asymetry in the number of pixels vs. the number of acquistioins in Sentinel-1 time series (e.g. 100,000 pixels to 200 times) strongly favours sICA, but in the case that deformation and a topographically APS are not spatially independent as is often found at stratovolcanoes, tICA can be used.  

This readme contains a brief summary of the tunable parameters, and two examples (sICA and tICA).  

# Settings

The key settings are:
- figure_type - window to get interactive figures to explore the data, png to export when settings are finalised
- hdbscan and tsne parameters - can be changed via the sliders in the interactive figure, or set here if exporting to png.  
- sica or tica - depending on where you're working.
- downsampling - high to make it fast while you explore the data, low and slow for final results.
- n_pca_comp_start (and stop) - PCA is used to downsample the data.  Explore how a low number can discard signal, but a high number can include noise.  Set a range, and then chose the best value in the interactive sources explorer.  


```python


def LiCSAtmo_correction(
        outdir,
        location,                                   # results are this directory in ourdir
        xy_list,                                    # time series of these pixels (list of tuples) is plotted.  Note, these are after downsampling.  
        automatic_selection = True,                 # if True, topographicallay correlated APS IC is selected automatically.  If False, manual choice.                                         
        licsbas_dir = None,                         # Option 1 for input data - licsbas time series.  
        licsbas_jasmin_dir = None,                  # Option 2 for input data - COMET LICSBAS Jasmin Volcano Portal directory 
        data_as_arg = None,                         # Option 3 for input data - Pass a dictionary.  see LiCSAtmo_correction() for details
        alignsar_dc = None,                         # Option 4 for input data - AlignSAR datacube.  
        licsalert_settings = None,                  # control downsampling using LiCSAlert functions, see below
        icasar_settings = None,                     # ICA settings, see below
        licsbas_settings = None,                    # If using a LiCSBAS time series, see below.   
        licsalert_pkg_dir = None,                   # Used to import some functions.  
        ):   


licsalert_settings = {"figure_type"         : 'both',             # either 'window' or 'png' (to save as pngs), or 'both'
                      "downsample_run"      : 0.3,                # data can be downsampled to speed things up
                      "downsample_plot"     : 0.5,                # and a 2nd time for fast plotting.  Note this is applied to the restuls of the first downsampling, so is compound
                      }

icasar_settings = {"sica_tica"              : 'tica',              # sica or tica
                   "n_pca_comp_start"       : 6,                   # pca is used as dimension reduction.  Number of PCs to start search at                               
                   "n_pca_comp_stop"        : 7,                   # and to stop search at.                                  
                   "bootstrapping_param"    : (200, 0),            # (number of runs with bootstrapping, number of runs without bootstrapping)
                    "hdbscan_param"         : (35, 10),            # Controls clustering of ICA sources, see HDBSCAN docs.    (min_cluster_size, min_samples)
                    "tsne_param"            : (30, 12),            # Controls 2D space representation, see tSNE docts.   (perplexity, early_exaggeration)
                    "ica_param"             : (1e-2, 150),         # FastICA settings.   (tolerance, max iterations)
                    "ifgs_format"           : 'cum',               #  can be 'all', 'inc' (incremental - short temporal baselines), or 'cum' (cumulative - relative to first acquisition)
                    "load_fastICA_results"  : True}                # If these exist, they will be loaded rather than recalculated

licsbas_settings = {"filtered"              : False,              # Boolean, True for filtered data
                    "date_start"            : None,                # crop time series in time
                    "date_end"              : None,                # crop time series in time
                    'mask_type'             : 'licsbas',           # "dem" or "licsbas"
                    'crop_pixels'           : None}                # crop pixels in space 
```

# Example 1: Campi Flegrei

They key figure is the interactive results explorer, here sICA at Campi Flegrei.  Every point is a source recovered by FastICA, projected into a 2D space (tSNE) and coloured by cluster (by HDBSCAN).  See Gaddes et al. 2019 for full details.  
Note that PC6 is greyed out as it's not being included.  IC0 is deformation.  

<img width="1878" height="977" alt="04_clustering_and_manifold_results" src="https://github.com/user-attachments/assets/99ec87da-f027-45ea-aeb8-25567b82a73a" />

ICs in space and time.  IC0 looks like deformation, and IC1 has a cumulative  time course that looks somewhat seasonal.  
<img width="1400" height="800" alt="03_ICA_sources_time" src="https://github.com/user-attachments/assets/9d32448f-45cb-495a-a869-f7d853f8de35" />

Comparison of the ICs with the DEM, and their use in time.  IC0 is not correlated with the DEM, but is used linearly in time so is deformation.  IC1 is very correlated with the DEM, but weakly in time so is a topographically correlated APS.  
<img width="1500" height="700" alt="03_ICA_sources_correlations" src="https://github.com/user-attachments/assets/fd724df9-ae56-415a-856d-a29a021b818d" />

Time series after we discard IC1 (topo. correlated APS).  Minimal change over the full time series, except over a region of high toporaphy.  
<img width="1200" height="400" alt="original_reconstruction_residual" src="https://github.com/user-attachments/assets/6d3599f1-58c4-4ae9-a0a4-6250bedace19" />

Time series for a pixel in the area of high topography.  Top left: original data, with smoothed line.  Top right: corrected.  Centre: smoothed comparison, showing the seasonal signal removed (although with a slight change in the cumulative displacement of 1-2cm over 7 years).  
<img width="1200" height="700" alt="original_reconstructed_ts_0" src="https://github.com/user-attachments/assets/322f168b-a4a1-4aa2-829f-a31014cd8b68" />

# Example 2: Vesuvius
Interactive results explorer for Vesuvius with tICA (note that spatial signals from the sICA example are now time signals.  Deformation is visible in IC0, and IC1 and IC4 look like a seasonal signal. 
<img width="1878" height="977" alt="04_clustering_and_manifold_results" src="https://github.com/user-attachments/assets/1f8b5c91-3d54-4c19-8e06-c21a48ed5a65" />

ICs in space and time.  IC0 looks like deformation in time, and IC1 and 4 have a slight seasonal signal in time.  
<img width="1400" height="800" alt="03_ICA_sources_time" src="https://github.com/user-attachments/assets/4d4b7be0-8ab5-43b8-b766-97e30977f30c" />

Comparison of the ICs with the DEM, and their use in time.  IC0 is linear in time, but also correlated with the DEM.  However, this is what we'd expect at a stratovolcano.  ICs 1 and 4 are also closely correlated with the DEM, but not in time.  Let's manually choose both as the signals to remove for the correction.  
<img width="1500" height="700" alt="03_ICA_sources_correlations" src="https://github.com/user-attachments/assets/3a9fac33-5c03-4702-84b2-0d9761e5da67" />

The time series after IC1 and 4 are removed, showing little change in the cumulative displacement obesrved.  
<img width="1200" height="400" alt="original_reconstruction_residual" src="https://github.com/user-attachments/assets/27904fa8-a87f-4d16-badb-63fa14d03d12" />

ATime series for a pixel in the area of high topography and deformaing region.  As per sICA at Campi Flegrei, top left shows the original data, which we interpret as subsidence with a seasonal signal.  Top right: LiCSAtmo correction, showing subsidence (which agrees with the GPS-derived displacements - see full paper), and bottom: comparison of the smoothed data.  
<img width="1200" height="700" alt="original_reconstructed_ts_0" src="https://github.com/user-attachments/assets/af3b8d10-0634-403b-b4f7-0e9a1faa052e" />
