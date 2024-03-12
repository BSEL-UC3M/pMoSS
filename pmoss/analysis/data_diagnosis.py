# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 19:39:28 2018

@author: egomez
"""

import numpy as np
import pandas as pd
import os
import shutil
from pmoss.utils import create_combination
from pmoss.loaders import morphoparam
from pmoss.statistics import cross_validated_pvalues
from pmoss.models import decission_data_lowess, decission_data_exponential

def get_decision_index(decission_param, data_features, combination_dict):
    Theta = pd.DataFrame()
    for c in range(len(combination_dict)):
        Theta_2 = pd.DataFrame()
        Theta_2['comparison'] = [combination_dict[str(c)]]
        for m in range(len(data_features)):
            aux = decission_param[decission_param.comparison == combination_dict[str(c)]][
                data_features[str(m)] + '_convergence_d']
            aux = aux.values[0]
            if ~np.isnan(float(aux)):
                Theta_2[data_features[str(m)] + ' Theta'] = [float(aux >= 0)]
            else:
                Theta_2[data_features[str(m)] + ' Theta'] = [0]
            Theta_2[data_features[str(m)] + ' Theta'] = Theta_2[data_features[str(m)] + ' Theta'].astype(int)
        frames = [Theta, Theta_2]
        Theta = pd.concat(frames)

    return Theta

def data_diagnosis(file_name, gamma, alpha, grid_size, n0,Nmax,k,
                   initial_portion,path=None,method = None, test = None, 
                   path2images=None, group_dict = None, temp_folder='../computed_pvalues/'):
    
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
    if not os.path.exists(temp_folder):
            try:
                os.makedirs(temp_folder)
            except OSError as e:
                print ("Error: %s - %s." % (e.filename, e.strerror))
                print ("{} folder could not be created".format(temp_folder))
        
    
    # Estimate p-values    
    df_pvalues = cross_validated_pvalues(data, data_features, group_dict, 
                                         grid_size, n0, Nmax, k, 
                                         initial_portion, test = test, temp_folder=temp_folder)
    
    # Save p-values in the temporary directory
    np.save(os.path.join(temp_folder, 'all_pvalues.npy'), df_pvalues)
    
    
    # Compute the decision analysis of the estimated p-values
    if method == 'lowess':
        decission_param = decission_data_lowess(df_pvalues, combination_dict,
                                                data_features,
                                                sign_level = alpha,
                                                gamma = gamma)
        
    elif method == 'exponential' or method is None:
        decission_param = decission_data_exponential(df_pvalues, 
                                                     combination_dict, 
                                                     data_features, 
                                                     sign_level = alpha, 
                                                     gamma = gamma)
    
    # Calculate the decision index
    Theta = get_decision_index(decission_param, data_features, 
                               combination_dict)
    
    return df_pvalues, decission_param, Theta

def compute_diagnosis(file_name, path=None,
                      gamma=None, alpha=None, grid_size=None, n0=None,
                      Nmax=None, k=None, initial_portion=None, method=None,
                      test=None, group_dict=None, output_folder='../computed_pvalues/'):
    """
    This function initializes all the parameters and computes the data diagnosis.
    See the manuscript for further detail.
    """
    # Define the default parameters for a regular data diagnosis.
    if path is None:
        path = os.getcwd() + '/'

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
        initial_portion = 1 / 3.

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
                                                        grid_size, n0, Nmax,
                                                        k, initial_portion,
                                                        path=path,
                                                        method=method,
                                                        test=test,
                                                        group_dict=group_dict,
                                                        temp_folder=output_folder)

    # Remove the temporary directory
    mydir = output_folder
    ## Try to remove tree; if failed show an error using try...except on screen
    try:
        shutil.rmtree(mydir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    return df_pvalues, decission_param, Theta
