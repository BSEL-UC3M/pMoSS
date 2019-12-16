# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 19:39:28 2018

@author: Estibaliz Gómez de Mariscal
"""
import os
import shutil

from data_diagnosis import data_diagnosis

def compute_diagnosis(file_name,path = None, 
                      gamma = None, alpha = None, grid_size = None, n0 = None,
                      Nmax = None, k = None, initial_portion = None,method=None, 
                      test = None, group_dict = None ):
    """
    This function initializes all the parameters and computes the data diagnosis.
    See the manuscript for further detail.
    """
    # Define the default parameters for a regular data diagnosis. 
    if path is None:
        path = os.getcwd()+'/'
        
    if gamma is None:
        gamma = 5e-06
        
    # Satistical significance level. Defaul 95%
    if alpha is None:
        alpha = 0.05
        
    if grid_size is None:
        grid_size = 200
        
    if n0 is None:
        n0 = 20        
        
    if Nmax is None: 
        Nmax = 2500
        
    if k is None:
        k = 20
        
    if initial_portion is None:
        initial_portion = 1/3.
        
    if test is None:
        test = 'MannWhitneyU'
        
    if method is None:
        method = 'exponential'
        
#    if group_dict is None and file_name is None:
#        """
#        This option is not recommended. It only works with the data about 
#        cellular protrusions.
#        """
#        df_pvalues, decission_param, Theta = data_diagnosis(gamma, alpha, 
#                                                            grid_size, n0,Nmax,
#                                                            k, initial_portion,
#                                                            path=path,
#                                                            test = test,
#                                                            method=method)
#    elif file_name is None:
#        """
#        This option is not recommended. It only works with the data about 
#        cellular protrusions.
#        """
#        df_pvalues, decission_param, Theta = data_diagnosis(gamma, alpha, 
#                                                            grid_size, n0,Nmax,
#                                                            k, initial_portion,                                                            
#                                                            path=path,
#                                                            test = test,
#                                                            group_dict=group_dict,
#                                                            method=method)
#    else:
    """
    Provide always the file_name
    """
    df_pvalues, decission_param, Theta = data_diagnosis(file_name, gamma, alpha, 
                                                            grid_size, n0,Nmax,
                                                            k, initial_portion,                                                            
                                                            path=path,
                                                            method=method,
                                                            test = test,
                                                            group_dict=group_dict)
        

    # Remove the temporary directory
    mydir= '../computed_pvalues/'
    ## Try to remove tree; if failed show an error using try...except on screen
    try:
        shutil.rmtree(mydir)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
        
    return df_pvalues, decission_param, Theta

        

                                                          
                                                          
