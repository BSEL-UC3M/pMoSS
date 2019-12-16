# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:46:06 2019

@author: egomez
"""
import pandas as pd
import numpy as np
import os

def load_pvalue_data(name, path = None):
    if path is None:
        path = os.getcwd()
    if name == 'cell morphology':
        measure = {
                          '0': 'Cell body size microns',
                          '1': 'Cell body perimeter microns',
                          '2': 'Cell body flatness',
                          '3': 'Cell body roundness',
                          '4': 'Cell body axis ratio',
                          '5': 'protrusion_number'
                        }
        test = {  '0': 'MannWhitneyU',
                  '1': 't-test',
                }
#            cell_morphology = np.load('morphological_measures_p_value_new.npy')
        cell_morphology = np.load(path + 'morphological_measures_p_value_new_total.npy')
        df = pd.DataFrame()
        df['p_value'] = cell_morphology[:,0]
        df['measure'] = cell_morphology[:,1]
        df['N'] = cell_morphology[:,2]
        df['comparison'] = cell_morphology[:,3] 
        df['test'] = cell_morphology[:,4]
        del cell_morphology
    elif name == 'protrusion_morphology':
        measure = {
                          '0': 'area_mu**2',
                          '1': 'perimeter_mu',
                          '2': 'length',
                          '3': 'diameter'
                        }
        test = {  '0': 'MannWhitneyU',
                  '1': 't-test',
                }
        protrusion_morphology = np.load(path + 'p_values_prot_morpho_new.npy')
        df = pd.DataFrame()
        df['p_value'] = protrusion_morphology[:,0]
        df['measure'] = protrusion_morphology[:,1]
        df['N'] = protrusion_morphology[:,2]
        df['comparison'] = protrusion_morphology[:,3] 
        df['test'] = protrusion_morphology[:,4]
        del protrusion_morphology
    elif name == 'protrusion_binary':
        measure = {  '0': 'protrusion_binary'}
        test = {  '0': 'ChiSquared'}
        protrusions_binary = np.load(path + 'p_values_prot_number_binary_new.npy')
        df = pd.DataFrame()
        df['p_value'] = protrusions_binary[:,0]   
        df['N'] = protrusions_binary[:,1] 
        df['comparison'] = protrusions_binary[:,2]   
        df['test'] = 'ChiSquared'
        df['measure'] = 'protrusion_binary'
        del protrusions_binary
    return df, measure, test