# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:33:58 2019

@author: egomez
"""
import numpy as np
import pandas as pd
import statsmodels.api

def significance_analysis(df, sign_level = None):
    if sign_level is None:
        sign_level = 0.05
    df['N'] = df['N'].astype('category')
    df['p_value'] = df['p_value'].astype('float')
    
    # Mean and standard error of the p-values
    mean = df.groupby('N')['p_value'].mean()
    yerr = df.groupby('N')['p_value'].sem()     
    
    ymax = mean + yerr
    x = np.array(mean.index)
    x = x.astype(int)
    # Save the data to fit a lowess regression
    dy = np.transpose(np.vstack((x,mean)))
      
    # Obtain the N such that all p-values are smaller than 0.05 (Nalpha in the manuscript)
    Nsign = x[(ymax-sign_level) <= 0]
    if len(Nsign) > 0:
        Nsign = Nsign[0]
    if Nsign == []:
        Nsign = 'NaN'


    # Fit a LOWESS model to the data.
    L = statsmodels.api.nonparametric.lowess(dy[:,1] , dy[:,0], frac=1./3, it = 5)
    area_under_curve = np.trapz(L[:,1], x = L[:,0])
    ref_area = (np.max(x)-np.min(x))*sign_level                        
    d = np.round(ref_area - area_under_curve)
    return L, d, Nsign

def convergence_analysis(L, gamma = None, sign_level = None):
    
    # Calculate the derivative of the LOWESS fit L. 
    # L = np.unique(L, axis = 0)
    
    aux = np.array([np.unique(L[:,0]),
               np.unique([L[L[:,0]==i,1] for i in L[:,0]])])
    L = np.transpose(aux, [1,0])
    
    x = L[:,0]
    y = L[:,1]
    dy = np.zeros(y.shape,float)
    dy[0:-1] = np.diff(y)/np.diff(x)
    dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    
    if gamma is None:
        gamma = 5e-06
       
    if sign_level is None:
        sign_level = 0.05

    area_under_curve = np.trapz( L[np.abs(dy) > gamma,1], x = L[np.abs(dy) > gamma,0])
    if len(x[np.abs(dy) > gamma]) > 0:
        convergence_N = np.max(x[np.abs(dy) > gamma])
        ref_area = (convergence_N - np.min(x))*sign_level
    else:
        convergence_N = 'NaN'
        ref_area = (x[-1]- np.min(x))*sign_level
    d = np.round(ref_area - area_under_curve)
      
    return d, convergence_N
        
def decission_data_lowess(df, combination_dict, data_features, sign_level = None, gamma = None):
    
    decission_param = pd.DataFrame()
       
    if sign_level is None:
        sign_level = 0.05
    if gamma is None:
        gamma = 5e-06
            
    for c in range(len(combination_dict)): 
        df_comparison = df[df.comparison == combination_dict[str(c)]]
        aux = pd.DataFrame()
        aux['comparison'] = [combination_dict[str(c)]]
        
        for i in range(len(data_features)):
            if data_features[str(i)] == 'protrusion_binary':
                aux['test'] = 'ChiSquared'
            else:
                aux['test'] = 'MannWhitneyU'
            df_measure = df_comparison[df_comparison.measure == data_features[str(i)]]
            L, d, Nsign = significance_analysis(df_measure, sign_level = sign_level)

            aux[data_features[str(i)] + '_nalpha_estimated'] = [Nsign]
            aux[data_features[str(i)] + '_d'] = [d]

            if d > 0:
                convergence_d, convergence_N = convergence_analysis(L, gamma = gamma, sign_level = sign_level)
                aux[data_features[str(i)] + '_convergence_N'] = [convergence_N]
                aux[data_features[str(i)] + '_convergence_d'] = [convergence_d]
            else:
                aux[data_features[str(i)] + '_convergence_N'] = ['NaN']
                aux[data_features[str(i)] + '_convergence_d'] = ['NaN']

                    # print(combination[str(c)])
        decission_param = pd.concat([decission_param, aux])
    return decission_param