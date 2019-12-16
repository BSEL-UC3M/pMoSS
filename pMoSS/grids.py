# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 19:39:28 2018

@author: egomez
"""
import numpy as np
import pandas as pd

#def filter_dataframe(df, col_name, values):
    # This function removes the the rows in df whose corresponding 'col_name' category is not in the dictionary 'values'
#    
#    df_new = pd.DataFrame()
#    for v in range(len(values)):
#        df_aux = df[df[col_name] == values[np.str(v)]]
#        df_new = pd.concat([df_new,df_aux])
#    return df_new


def get_datasize(df, group, group_dict):
    # group: name of the variable to measure
    # group_dict: dictionary with the different classes/groups by the variable group.
    
    m = 0.
    for  c in range(len(group_dict)):
        aux = df[df[group]==group_dict[np.str(c)]][group]
        # if len(aux) < m:
        if len(aux) > m:
            m = len(aux)
            
    return m

def get_grids(n0, Nmax, m, grid_size = None, k = None, initial_portion = None ):
    # n0: minimum value in the grid of n
    # Nmax: maximum value in the grid of n
    
    # m: data size to consider for the amount of k-fold in the cross validation
    # k: weight to determine the amount of k-folds when n = Nmax
    # initial_portion: weight to limit the amount of k-folds when n = n0
    
    # Default parameters
    if grid_size is None:
        grid_size = 250
    if initial_portion is None:
        initial_portion = 1/3.
    if k is None: 
        k = 20
        
    # Grid calculation
#     if m < Nmax:
#         Nmax = m
    grid_n = np.exp(np.linspace(np.round(np.log(n0)), np.log(Nmax), grid_size))
    grid_n = grid_n.astype(np.int)
    grid_n = np.unique(grid_n)  
    
    # folds i calculation from the grid
    final_fold = k*(m/min(m,np.max(grid_n)))       
    final_fold = np.int(final_fold)        
    folds = np.exp(np.linspace(np.log((m/n0)*initial_portion), np.log(final_fold),  len(grid_n)))     
    folds = folds.astype(np.int)

    return grid_n, folds  
