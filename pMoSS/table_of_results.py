# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:43:30 2019

@author: egomez
"""
# Calculate the decision index
import numpy as np    
from decision_index import get_decision_index
import pandas as pd
    
def table_of_results(param, data_features, combination_dict):
    Theta = get_decision_index(param, data_features, combination_dict)

    for i in range(len(data_features)):  
        t = Theta[['comparison', data_features[np.str(i)] + ' Theta']]
        p = param[['comparison', data_features[np.str(i)] + '_nalpha_estimated',
                   data_features[np.str(i)] + '_nalpha_theory']]  
        p=p.assign(a=np.nan,c=np.nan)
        p[['a','c']]=param[data_features[np.str(i)] + '_exp_params'].apply(lambda x: pd.Series([x[0], x[1]], index=['a', 'c']))
        p = p[['comparison', 'a', 'c', 
               data_features[np.str(i)] + '_nalpha_estimated',
               data_features[np.str(i)] + '_nalpha_theory']]
        p.rename(columns={'a': data_features[np.str(i)] + ' a', 
                        'c': data_features[np.str(i)] + ' c', 
                        data_features[np.str(i)] + '_nalpha_estimated': 
                        data_features[np.str(i)] + ' ^n-alpha',
                        data_features[np.str(i)] + '_nalpha_theory': 
                        data_features[np.str(i)] + ' n-alpha'},  
                        inplace=True)
    
        if i == 0:
            table = pd.merge(p, t, on='comparison')
        else:
            aux =  pd.merge(p, t, on='comparison')
            table = pd.merge(table,aux, on='comparison')
    return table
        
        