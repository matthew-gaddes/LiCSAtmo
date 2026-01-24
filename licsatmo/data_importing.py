#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 21:09:49 2026

@author: matthew
"""


def import_insar_data(
        volcano, volcano_dir, region, 
        licsalert_settings,
        icasar_settings, 
        licsbas_settings,
        licsbas_jasmin_dir = None,
        licsbas_dir = None,
        alignsar_dc = None,
        data_as_arg = None
        ):
    """
    Open InSAR data for LiCSAlert.  Data can either be a:
        licsbas_jasmin_dir - the path a a COMET volcano portal .json
        licsbas_dir  - the path to a LiCSBAS time series 
        alignsar_dc - an AlignSAR data cube.  
        data_as_arg - data processed in a different way.  
        
    Inputs:

        volcano | str | name of volcano frame. 
        volcano_dir | Path | outdir for volcano
        region | string | region that volcano lies in.  Optional?  
        
        licsalert_settings | dict
        icasar_settings  | dict 
        licsbas_settings | dict 
        
        licsbas_jasmin_dir | path |  COMET volcano portal
        licsbas_dir | path | 
        alignsar_dc
        data_as_arg | dict |
    
    Returns:
        displacement_r2 | dict, things like displacement, mask, DEM, lons etc.  
        
    History:
        2025_02_21 | MEG | Written.  
    
    """
    

    from licsalert.data_importing import LiCSBAS_to_LiCSAlert
    from licsalert.data_importing import LiCSBAS_json_to_LiCSAlert
    from licsalert.data_importing import AlignSAR_to_LiCSAlert
    

    
    # 3: Open the the input data, which can be in various formats (3 so far), 
    #    And and input settings for that type of data
    
    # 3.1: a JASMIN dir and associated text file of settings.  
    if licsbas_jasmin_dir is not None:
        # read various settings from the volcano's config file

        
        
        print(f"LiCSAlert is opening a JASMIN COMET Volcano Portal timeseries"
              " json file.  ")
        products = LiCSBAS_json_to_LiCSAlert(
            licsbas_jasmin_dir / region / f"{volcano}.json",
            licsbas_settings['crop_side_length'], 
            licsbas_settings['mask_type']
            )          
        
        (displacement_r3, tbaseline_info, ref_xy, 
          licsbas_json_creation_time) = products
        


    # remaining two ways to pass data to function.  
    else:
        # check user has provided inputs.  
        check_required_args(
            licsalert_settings, 
            ['figure_type', 
             'downsample_run',
             'downsample_plot',
             ],
            'licsalert_settings',
            )    
    
        check_required_args(
            icasar_settings, 
            ['n_pca_comp_start',
             'n_pca_comp_stop',
             ],
            'icasar_settings'
            )

        # 3.2: As a LiCSBAS direcotry
        if licsbas_dir is not None:
            # check licsbas_settings are provided.  
            check_required_args(licsbas_settings, ['mask_type', 'filtered'],
                                'licsbas_settings')
            print(f"LiCSAlert is opening a LiCSBAS directory.  ")
            # if there are no licsbas settings, set them to the default.  
            if licsbas_settings == None:
                licsbas_settings = {"filtered"              : False,
                                    "date_start"            : None,
                                    "date_end"              : None,
                                    'mask_type'             : 'licsbas',
                                    'crop_pixels'           : None}
            displacement_r3, tbaseline_info = LiCSBAS_to_LiCSAlert(
                licsbas_dir,
                figures=True,
                n_cols=5,                              
                **licsbas_settings,
                )

        elif alignsar_dc is not None:
            print(f"LiCSAlert is opening a an AlignSAR data cube.  ")
            ts = AlignSAR_to_LiCSAlert(alignsar_dc, )
            displacement_r3, tbaseline_info = ts
            

        # 3.3: Data processed with users own approach/software.  
        else:
            print(f"LiCSAlert is using data that was passed to the function as an argument  ")
            displacement_r3 = data_as_arg['displacement_r3']
            tbaseline_info = data_as_arg["tbaseline_info"]
            if licsbas_settings is not None:
                print(f"'licsbas_settings' can only be provided if a "
                      f"licsbas_dir is provided as the input data.  As data "
                      f"is being passed to LiCSAlert in a different way, "
                      f"licsbas_settings will be removed.  ")
                del licsbas_settings
                
    # however data is used, these two arguments must agree.  
    icasar_settings['figures'] = licsalert_settings['figure_type']   
        

    return displacement_r3, tbaseline_info
    

#%%

def check_required_args(settings_dict, required_inputs, settings_name):
    """ Check that input dictionary contains the required keys.  
    Inputs:
        settings_dict | dict | contains keys and values of the setting.  
        required_inputs | list | keys that must be in dict. 
        settings_name | str | name of dict, to make error messages clearer
    Returns:
        exception if required is missing.  
    History:
        2024_01_26 | MEG | Written.  
    """
    
   
    if settings_dict is None:
        raise Exception(f"'{settings_name}' is None, but should be a dictionary "
                        f"of settings (see the examples).  Exiting.  ")
    
    for required_input in required_inputs:
        if not (required_input in settings_dict.keys()):
            raise Exception(f"'{required_input}' was not found in "
                            f"'{settings_name}', and it is not optional.  "
                            f"Exiting.  ")
    print(f"All the required arguments were found in the dictionary.  ")
