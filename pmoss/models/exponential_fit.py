# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:02:11 2019

@author: egomez
"""
######################
## Compute the decision analysis of the estimated p-values
#decission_param = decission_data(df_pvalues, combination_dict, data_features, sign_level = alpha, gamma = gamma)
#
#
## Calculate the decision index
#Theta = get_decision_index(decission_param, data_features, combination_dict)
######################
import numpy as np
import pandas as pd
import scipy


def func_exp_pure(x, a, c):    
    return a*np.exp(-c*(x))

def n_gamma_function(gamma_par, a, c):
    gamma_par = np.array(gamma_par, dtype=float)
    a = np.array(a, dtype=float) 
    c = np.array(c, dtype=float)        
    return  np.floor((-1/c)*np.log((gamma_par)/(c*a)))


def distance(alpha_par, n, a, c):
    n = np.array(n, dtype=float)
    alpha_par = np.array(alpha_par, dtype=float) 
    a = np.array(a, dtype=float) 
    c = np.array(c, dtype=float)       
    A_alpha = alpha_par*n 
    A = ((1./c)*a)*(1-np.exp(-n*c))
    return A_alpha - A


def nalpha_theory(a, c, sign_level = None):
    """
    This function solves the equation alpha = aexp(-cn) for n. The solution 
    might not be a valid solution in the following cases:
    
     -  such an n value might not exists as aexp(-cn) is almost constant with 
        a>alpha
     -  aexp(-cn) is always smaller than alpha, so we set n = 0 
    """
    if sign_level is None:
        sign_level = 0.05
    a = np.array(a, dtype=float) 
    c = np.array(c, dtype=float) 
    if a <= sign_level:
        nalpha = 0
    else:
        nalpha = np.floor((-1/c)*np.log(sign_level/a)) 
    if nalpha < 0 and a <= sign_level:
        nalpha = 0
    elif nalpha < 0 and a > sign_level:
        nalpha = np.inf
    return nalpha

def nalpha_estimate(df, sign_level = None):
    """
    This function provides an estimation of n-alpha taking into account the 
    bias of the sample, i.e. the distribution of the p-values for each value of
    n. 
    n-alpha = argmin_n(| mean(p(n)) + std(p(n)) | < alpha)
    n-alpha might not exist for two different reasons:
        - the p-values are uniformly distributed above alpha so this n-value 
        will never exist.
        - There is not enough data, i.e. sample size < n-alpha, so we do not 
        have enough p-values for the assessment of n-alpha
    """
    if sign_level is None:
        sign_level = 0.05
    # Mean and standard error of the p-values
#    df['p_value'] = pd.to_numeric(df['p_value'], downcast='float')
    df = df.astype({"p_value": float})
    mean = df.groupby('N')['p_value'].mean()
    yerr = df.groupby('N')['p_value'].sem()     
    
    ymax = mean + yerr
    x = np.array(mean.index)
    x = x.astype(int)
    # Obtain the n-value such that all p-values are smaller than 0.05
    # (n-alpha in the manuscript)
    Nsign = x[(ymax-sign_level) <= 0]
    if len(Nsign) > 0:
        # This value exists and we have enough samples as to assess it.
        Nsign = Nsign[0]
    else:
        # Either the value does not exist or we do not have enough data to assess it.
        Nsign = np.nan
#    if Nsign == []:
#        Nsign = 'NaN'
    return Nsign
        
def decission_data_exponential(df, combination_dict, data_features, sign_level = None, gamma = None):
    if sign_level is None:
        sign_level = 0.05
    if gamma is None:
        gamma = 5e-06

    decission_param = pd.DataFrame()
            
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

            if np.sum(df_measure['p_value'].astype(float)>sign_level)>0:
                # Get the parameters of an exponential fit
                pop,_ = scipy.optimize.curve_fit(func_exp_pure,df_measure['N'].astype(float),
                                                 df_measure['p_value'].astype(float),
                                                 method = 'trf',
                                                 bounds=([0.,0.],[1., np.inf]))
            
                # Get the convergence point of the exponential function
                convergence_N = n_gamma_function(gamma, pop[0], pop[1])
                # The function aexp(-cn) is differentiable, which means that its
                # derivative is continuous and indeed, it is a decreasing 
                # function. If gamma = |(aexp(-cn))'| = acexp(-cn) does not 
                # exist, then a*c is already smaller than gamma and hence, we 
                # can accept that aexp(-cn) is constant. Hence:
                if convergence_N<0 and  pop[0] <= sign_level:
                    # If the exponential is almost constant and a<alpha, there
                    # exists statistical significance, so convergence_N = 1.
                    convergence_N = 1
                    convergence_d = distance(sign_level, convergence_N, pop[0], pop[1])
                elif convergence_N < 0:
                    # If the exponential is almost constant and a>alpha, there 
                    # will never be statistical significance.
                    convergence_N = np.inf
                    convergence_d = -np.inf
                    
                else:
                    # Get the difference between area under the exponential 
                    # curve and the constant function alpha.
                    convergence_d = distance(sign_level, convergence_N, pop[0], pop[1])
                if pop[0]*np.exp(-pop[1]*50)<sign_level:
                    # The case in which the p-value is already less than alpha 
                    # for very few samples.
                    convergence_d = np.inf
            else:
                convergence_N = 0
                convergence_d = np.inf
                pop = [0.,0.] # The exponential function is equivalen to the null function. 
            # Store the values as a dataframe
            aux[data_features[str(i)] + '_exp_params'] = [pop]
            aux[data_features[str(i)] + '_nalpha_estimated'] = [nalpha_estimate(df_measure, sign_level = None)]
            aux[data_features[str(i)] + '_nalpha_theory'] = [nalpha_theory(pop[0], pop[1], sign_level = None)]
            aux[data_features[str(i)] + '_convergence_N'] = [convergence_N]
            aux[data_features[str(i)] + '_convergence_d'] = [convergence_d]

        decission_param = pd.concat([decission_param, aux])
    return decission_param
