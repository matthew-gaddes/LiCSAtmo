#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 18:54:27 2026

@author: matthew
"""



def r3_to_r2(phUnw):
    """ Given a rank3 of ifgs, convert it to rank2 and a mask.  
    Works with either masked arrays or just arrays.  
    Inputs:
        phUnw | rank 3 array | n_ifgs x height x width
    returns:
        r2_data['ifgs'] | rank 2 array | ifgs as row vectors
        r2_data['mask'] | rank 2 array 
    History:
        2020/06/09 | MEG  | Written
        2024_08_25 | MEG | Add to LiCSAlert
    """
    import numpy as np
    import numpy.ma as ma
    
    # if it's a masked array, get the number of non-masked pixels
    if ma.isMaskedArray(phUnw):
        n_pixels = len(ma.compressed(phUnw[0,]))                                          
        mask = ma.getmask(phUnw)[0,]                                                        
    # or if a normal numpy array, just get the number of pixels
    else:
        n_pixels = len(np.ravel(phUnw[0,]))                                                 
        # no mask (all are valid)
        mask = np.zeros(phUnw[0,].shape)                                                    
 
    # initiate to store ifgs as rows in
    r2_ifgs = np.zeros((phUnw.shape[0], n_pixels))                                          
    for ifg_n, ifg in enumerate(phUnw):
        # non masked pixels into row vectors
        if ma.isMaskedArray(phUnw):
            r2_ifgs[ifg_n,] = ma.compressed(ifg)                                            
        # or all just pixles into row vectors
        else:
            r2_ifgs[ifg_n,] = np.ravel(ifg)                                                 

    r2_data = {'ifgs' : r2_ifgs,                                                            
               'mask' : mask}          
    return r2_data




def r2_to_r3(ifgs_r2, mask):
    """ Given a rank2 of ifgs as row vectors, convert it to a rank3.   Copied from insar_tools.general to avoid making insar_tools a dependency.  
    Inputs:
        ifgs_r2 | rank 2 array | ifgs as row vectors 
        mask | rank 2 array | to convert a row vector ifg into a rank 2 masked array        
    returns:
        phUnw | rank 3 array | n_ifgs x height x width
    History:
        2020/06/10 | MEG  | Written
    """
    import numpy as np
    import numpy.ma as ma
        
    n_ifgs = ifgs_r2.shape[0]
    ny, nx = col_to_ma(ifgs_r2[0,], mask).shape                                   # determine the size of an ifg when it is converter from being a row vector
    
    ifgs_r3 = np.zeros((n_ifgs, ny, nx))                                                # initate to store new ifgs
    for ifg_n, ifg_row in enumerate(ifgs_r2):                                           # loop through all ifgs
        ifgs_r3[ifg_n,] = col_to_ma(ifg_row, mask)                                  
    
    mask_r3 = np.repeat(mask[np.newaxis,], n_ifgs, axis = 0)                            # expand the mask from r2 to r3
    ifgs_r3_ma = ma.array(ifgs_r3, mask = mask_r3)                                      # and make a masked array    
    return ifgs_r3_ma


def col_to_ma(col, pixel_mask):
    """ A function to take a column vector and a 2d pixel mask and reshape the column into a masked array.  
    Useful when converting between vectors used by BSS methods results that are to be plotted
    
    Inputs:
        col | rank 1 array | 
        pixel_mask | array mask (rank 2)
        
    Outputs:
        source | rank 2 masked array | colun as a masked 2d array
    
    2017/10/04 | collected from various functions and placed here.  
    
    """
    import numpy.ma as ma 
    import numpy as np
    
    source = ma.array(np.zeros(pixel_mask.shape), mask = pixel_mask )
    source.unshare_mask()
    source[~source.mask] = col.ravel()   
    return source