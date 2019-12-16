# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 19:39:28 2018

@author: egomez
"""
import numpy as np
import pandas as pd

def get_decision_index(decission_param, data_features, combination_dict):
    Theta = pd.DataFrame()
    for c in range(len(combination_dict)):
        Theta_2 = pd.DataFrame()
        Theta_2['comparison'] = [combination_dict[np.str(c)]]
        for m in range(len(data_features)):
            aux = decission_param[decission_param.comparison == combination_dict[np.str(c)]][data_features[np.str(m)] + '_convergence_d']  
            aux = aux.values[0]
            if ~np.isnan(np.float(aux)):            
                Theta_2[data_features[np.str(m)] + ' Theta'] = [np.float(aux >= 0)]
            else:
                Theta_2[data_features[np.str(m)] + ' Theta'] = [0]
            Theta_2[data_features[np.str(m)] + ' Theta'] = Theta_2[data_features[np.str(m)] + ' Theta'].astype(np.int)
        frames = [Theta, Theta_2]
        Theta = pd.concat(frames)
    
    return Theta
                                                          
                                                          
