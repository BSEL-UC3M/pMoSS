# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:41:55 2019

@author: egomez
"""
import numpy as np
import pandas as pd
from ..models.exponential_fit import decission_data_exponential
from ..models.lowess_fit import decission_data_lowess
from ..analysis.data_diagnosis import get_decision_index

def get_Nmax(df, condition = None):
   if condition is None:
       condition = {'0': 'HD_control',
                  '1': 'HD_tax1',
                  '2': 'HD_tax50',
                  '3': 'LD_control',
                  '4': 'LD_tax1',
                  '5': 'LD_tax50'}
   Nmax = len(df)
   for  c in range(len(condition)):
       aux = df[df['Condition'] == condition[str(c)]]['Condition']
       if len(aux) < Nmax:
           Nmax = len(aux)
   return Nmax

def filter_dataframe(df, col_name, values):
    df_new = pd.DataFrame()
    for v in range(len(values)):
        df_aux = df[df[col_name] == values[str(v)]]
        df_new = pd.concat([df_new,df_aux])
    return df_new

def get_grids(n0, ninf, Nmax, nsize = None, k = None, initial_portion = None ):
    # n0: minimum value in the grid of n
    # Nmax: maximum value in the grid of n

    # m: data size to consider for the amount of k-fold in the cross validation
    # k: weight to determine the amount of k-folds when n = Nmax
    # initial_portion: weight to limit the amount of k-folds when n = n0

    # Default parameters
    if nsize is None:
        nsize = 250
    if initial_portion is None:
        initial_portion = 1/3.
    if k is None: 
        k = 20
    # Grid calculation
    grid_n = np.exp(np.linspace(np.round(np.log(n0)), np.log(ninf), nsize))
    grid_n = grid_n.astype(int)
    grid_n = np.unique(grid_n)  
    if len(grid_n) < nsize:
        while nsize-len(grid_n)>0:
            diff_grid_n = grid_n[1:]-grid_n[:-1]
#            a = grid_n[1:][diff_grid_n == np.max(diff_grid_n)][0]
#            b = grid_n[:-1][diff_grid_n == np.max(diff_grid_n)][0]
            index = np.random.randint(len(diff_grid_n))
            while diff_grid_n[index]==1:
                index = np.random.randint(len(diff_grid_n))
            a = grid_n[1:][index]
            b = grid_n[:-1][index]
            c = int(b + (a-b)/2)
            grid_n = np.insert(grid_n,-1,c)
            grid_n = np.unique(grid_n) 
    # folds i calculation from the grid
    final_fold = k*(Nmax/np.max(grid_n))
    final_fold = int(final_fold)
    folds = np.exp(np.linspace(np.log((Nmax*initial_portion)/n0), np.log(final_fold),  len(grid_n)))     
    folds = folds.astype(int)

    return grid_n, folds

def reduced_grids(grid_n, folds, nsize, portion, n0 = None, Nmax = None):
    if n0 is None:
        n0 = np.min(grid_n)
    if Nmax is None:
        Nmax = np.max(grid_n)
    
    aux_grid = grid_n[grid_n >= n0]
    aux_grid = aux_grid[aux_grid <= Nmax]   
    
    aux_folds = folds[grid_n >= n0]
    aux_folds = aux_folds[aux_grid <= Nmax]   
    
    reduced_grid = np.zeros(nsize)
    reduced_folds = np.zeros(nsize)
    
    cnte = np.linspace(0,len(aux_grid)-1, nsize)
    cnte = cnte.astype(int)
    for i in range(nsize):
        reduced_grid[i] = aux_grid[cnte[i]]
        reduced_folds[i] = int(aux_folds[cnte[i]]*portion)
    reduced_grid = reduced_grid.astype(int)
    reduced_folds = reduced_folds.astype(int)
    return reduced_grid, reduced_folds
       
def reduced_data(df, grid_n, folds, datatype = None, measure = None, comparison = None, test = None):
    # grid_n = new grid
    # folds = number of p-values to sample for each ni value in the grid. 
    # datatype = either cell, protrusions or prot_number
    # measure = name or list of names of the measures to take into account
    # test = name or series of the statistical tests to analyze    
    reduced_df = pd.DataFrame()
    if datatype == 'prot_number':
        for p in range(len(grid_n)):        
            n_sample = df[df.N == grid_n[p]].groupby('comparison').apply(lambda x: x.sample(n = folds[p]))
            reduced_df = [reduced_df, n_sample]
            reduced_df = pd.concat(reduced_df)
    else:        
        if test is None:
            test = {'0':'MannWhitneyU'}      
        if datatype == 'cell':
            if measure is None:
                measure = {'0': 'Cell body size microns',
                          '1': 'Cell body perimeter microns',
                          '2': 'Cell body axis ratio'}
        elif datatype == 'protrusions':
            if measure is None:
                measure = {'0': 'area_mu**2',
                          '1': 'perimeter_mu',
                          '2': 'length',
                          '3': 'diameter'}                
        for t in range(len(test)):
            df_aux = df[df.test == test[str(t)]]
            for m in range(len(measure)):
                df_aux_m = df_aux[df_aux.measure == measure[str(m)]]
                for p in range(len(grid_n)):        
                    n_sample = df_aux_m[df_aux_m.N == grid_n[p]].groupby('comparison').apply(lambda x: x.sample(n = folds[p]))
                    reduced_df = [reduced_df, n_sample]
                    reduced_df = pd.concat(reduced_df)             
          
    return reduced_df


def results_reduced_data(df, combination, measure, test, grid_n, folds, datatype, method=None):
    sign_level = 0.05
    gamma = 5e-06
    for i in range(0,100):
        reduced_df = reduced_data(df, grid_n, folds, datatype = datatype)
        
        # Compute the decision analysis of the estimated p-values
        if method == 'lowess':
            data_aux = decission_data_lowess(reduced_df, combination, measure, sign_level = sign_level, gamma = gamma)
            
        elif method == 'exponential' or method is None:
            data_aux = decission_data_exponential(reduced_df, combination, measure, sign_level = sign_level, gamma = gamma)
        
        # Calculate the decision index
        if i == 0:
            Theta = get_decision_index(data_aux, measure, combination)
        else:
            aux_theta = get_decision_index(data_aux, measure, combination)
            Theta = pd.concat([Theta,aux_theta])   
    result = np.zeros((len(measure), len(combination)))                
    for m in range(len(measure)):
        for c in range(len(combination)):
            aux = Theta[Theta['comparison'] == combination[str(c)]][measure[str(m)] + ' Theta']
            result[m,c] = np.mean(aux[0])*100
#        print(combination[str(c)])
#        print(measure[str(m)])
#        print(np.mean(aux[0]))
    return Theta, result