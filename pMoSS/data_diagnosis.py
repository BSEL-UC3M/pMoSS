# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 19:39:28 2018

@author: egomez
"""

import numpy as np
import os
from load_morphology_data import morphoparam
#from data_classification_labels import group_names
#from data_classification_labels import group_combination
from data_classification_labels import create_combination
from cross_validation import cross_validated_pvalues
from decision_index import get_decision_index


def data_diagnosis(file_name, gamma, alpha, grid_size, n0,Nmax,k,
                   initial_portion,path=None,method = None, test = None, 
                   path2images=None, group_dict = None):
    
    if test is None:
        test = 'MannWhitneyU'
        
    # Load data   
    data, data_features, group_labels = morphoparam(file_name, path=path)
        
#    if path2images is not None:
#        group_dict = group_names(path2images)
#        combination_dict = group_combination(path2images)
    if group_dict is None:
        group_dict = group_labels
        
    combination_dict = create_combination(group_dict)
    
    
    # Create a temporary directory to store the value during 
    # Monte-Carlo cross validation
    if not os.path.exists('../computed_pvalues/'):
            try:
                os.makedirs('../computed_pvalues/' )    
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))
                print ("../computed_pvalues/ folder could not be created")
        
    
    # Estimate p-values    
    df_pvalues = cross_validated_pvalues(data, data_features, group_dict, 
                                         grid_size, n0, Nmax, k, 
                                         initial_portion, test = test)
    
    # Save p-values in the temporary directory
    np.save('../computed_pvalues/' + 'all_pvalues.npy',df_pvalues)
    
    
    # Compute the decision analysis of the estimated p-values
    if method == 'lowess':
        
        from lowess_fit import decission_data_lowess
        decission_param = decission_data_lowess(df_pvalues, combination_dict,
                                                data_features,
                                                sign_level = alpha,
                                                gamma = gamma)
        
    elif method == 'exponential' or method is None:
        
        from exponential_fit import decission_data_exponential
        decission_param = decission_data_exponential(df_pvalues, 
                                                     combination_dict, 
                                                     data_features, 
                                                     sign_level = alpha, 
                                                     gamma = gamma)
    
    # Calculate the decision index
    Theta = get_decision_index(decission_param, data_features, 
                               combination_dict)
    
    return df_pvalues, decission_param, Theta


                                                          
                                                          
