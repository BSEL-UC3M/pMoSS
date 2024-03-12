# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 19:39:28 2018

@author: egomez
"""
import numpy as np
import scipy.misc
import scipy.io
import pandas as pd
import os
#from statistical_tests import obtain_pvalues_ChiSquared
#from statistical_tests import pvalues_continuous

def get_datasize(df, group, group_dict):
    # group: name of the variable to measure
    # group_dict: dictionary with the different classes/groups by the variable group.

    m = 0.
    for c in range(len(group_dict)):
        aux = df[df[group] == group_dict[str(c)]][group]
        # if len(aux) < m:
        if len(aux) > m:
            m = len(aux)

    return m

def get_grids(n0, Nmax, m, grid_size=None, k=None, initial_portion=None):
    # n0: minimum value in the grid of n
    # Nmax: maximum value in the grid of n

    # m: data size to consider for the amount of k-fold in the cross validation
    # k: weight to determine the amount of k-folds when n = Nmax
    # initial_portion: weight to limit the amount of k-folds when n = n0

    # Default parameters
    if grid_size is None:
        grid_size = 250
    if initial_portion is None:
        initial_portion = 1 / 3.
    if k is None:
        k = 20
    # Grid calculation
    grid_n = np.exp(np.linspace(np.round(np.log(n0)), np.log(Nmax), grid_size))
    grid_n = grid_n.astype(int)
    grid_n = np.unique(grid_n)

    # folds i calculation from the grid
    final_fold = k * (m / min(m, np.max(grid_n)))
    final_fold = int(final_fold)
    folds = np.exp(
        np.linspace(np.log((m / n0) * initial_portion), np.log(final_fold), len(grid_n))
            )
    folds = folds.astype(int)
    folds = np.sort(folds)[::-1]
    return grid_n, folds

def read_pvalues(file_list, temp_folder='../computed_pvalues/'):
    """

    :param file_list: a list with the names of the numpy arrays to read
    :param temp_folder: the folder where the files ar stored
    :return: a pandas array with the information of all the files is concatenated by rows.
    """
    df_pvalues = pd.DataFrame()
    for f in range(len(file_list)):
        aux = np.load(os.path.join(temp_folder, file_list[f]), allow_pickle=True)

        pd_aux = pd.DataFrame()
        pd_aux['p_value'] = aux[:, 0]
        pd_aux['N'] = aux[:, 1]
        pd_aux['comparison'] = aux[:, 2]
        pd_aux['test'] = aux[:, 3]
        pd_aux['measure'] = aux[:, 4]
        del aux

        df_pvalues = pd.concat([df_pvalues, pd_aux])
        del pd_aux

    return df_pvalues

def cross_validated_pvalues(df, data_features, group_dict, grid_size, n0, Nmax, k, initial_portion, test = None, temp_folder='../computed_pvalues/'):
    # df: dataframe containing numerical values of the different measures and ordered by groups 
    # group_dict: dictionary of the different groups
    # measure: dictionary of the measures 
    # N: subsample' size (ni of the grid for the N values). Each subsample has N number of observations
    # folds represent the number of times the experiment will be repeated (k-folds of the cross-validation)
    
    # Return
    # -----------------------------
    # df_p: (p.list) Dataframe containing all folds amount of p-values by measure and comparing the conditions two-by-two, for a specific amount of datapoints.
    # Initialize the dataframe
    if test is None:
        test = 'MannWhitneyU'
    m = get_datasize(df, 'Condition', group_dict)
    grid_n, folds = get_grids(n0, Nmax, m, grid_size = grid_size, k = k, initial_portion = initial_portion )
    
#    df_pvalues = pd.DataFrame()
    file_list = []
    # Compare each condition with the rest without repeating the comparisons        
    for c in range(len(group_dict)):
        if c+1 < len(group_dict):
            for k in range(c+1,len(group_dict)):
                df_pvalues = pd.DataFrame()

                sampleA =df[df.Condition==group_dict[str(c)]]
                sampleB =df[df.Condition==group_dict[str(k)]]
                l = min(len(sampleA), len(sampleB))
                
                grid_n_l = grid_n[grid_n<=l]
                folds_l = folds[grid_n<=l]
                print('comparison: ' + group_dict[str(c)] + '_' + group_dict[str(k)])
                for ni in range(len(grid_n_l)):
                    for epoch in range(folds_l[ni]):
                        
                        subA = sampleA.groupby('Condition').apply(lambda x: x.sample(n = grid_n_l[ni]))
                        subB = sampleB.groupby('Condition').apply(lambda x: x.sample(n = grid_n_l[ni]))

                        for m in range(len(data_features)):    
                            if data_features[str(m)] == 'protrusion_binary':
                                # Categorical variable and requires a different test.
                                observed_A = np.array([0,0])
                                observed_B = np.array([0,0])
                                # The data to compare has to be different 
                                while observed_A[0]+observed_B[0] == 0 or observed_A[1]+observed_B[1] == 0:
                                    subA = sampleA.groupby('Condition').apply(lambda x: x.sample(n = grid_n_l[ni]))
                                    subB = sampleB.groupby('Condition').apply(lambda x: x.sample(n = grid_n_l[ni]))
                                    auxA = subA['protrusion_binary'][group_dict[str(c)]]
                                    auxA = auxA.astype(float)
                                    auxB = subB['protrusion_binary'][group_dict[str(k)]]
                                    auxB = auxB.astype(float)
                                    # Portions of "1" and "0" are calculated.
                                    observed_A = np.array([ len(auxA) - sum(auxA) ,
                                                           sum(auxA),  len(auxA)])
                                    observed_B = np.array([ len(auxB) - sum(auxB) ,
                                                           sum(auxB),  len(auxB)])
                                    observed_A = 100*observed_A[0:2]/observed_A[2]
                                    observed_B = 100*observed_B[0:2]/observed_B[2]                                
                                    observed = pd.DataFrame([np.array(observed_A), np.array(observed_B)], 
                                                            index = [group_dict[str(c)], group_dict[str(k)]])

                                # Statistical test.
                                st, p, h0, expected = scipy.stats.chi2_contingency(observed = observed)
                                # Save the data.
                                df_p_aux = pd.DataFrame()
                                df_p_aux['p_value'] = [p]
                                df_p_aux['N'] = grid_n_l[ni]
                                df_p_aux['comparison'] = group_dict[str(c)] + '_' + group_dict[str(k)]
                                df_p_aux['test'] = 'ChiSquared'
                                df_p_aux['measure'] = data_features[str(m)]
                                frames = [df_pvalues, df_p_aux]
                                df_pvalues = pd.concat(frames)

                                
                            elif data_features[str(m)] != 'protrusion_number':
                                diff = 0.0
                                # Data to compare cannot be exactly equal
                                while diff == 0.0:
                                    subA = sampleA.groupby('Condition').apply(lambda x: x.sample(n = grid_n_l[ni]))
                                    subB = sampleB.groupby('Condition').apply(lambda x: x.sample(n = grid_n_l[ni]))
                                    auxA = subA[data_features[str(m)]][group_dict[str(c)]]
                                    auxA = auxA.astype(float)
                                    auxB = subB[data_features[str(m)]][group_dict[str(k)]]
                                    auxB = auxB.astype(float)
                                    diff = round(np.sum(auxB.values-auxA.values),3)
                                df_p_aux = pd.DataFrame()

                                if test == 'MannWhitneyU':
                                    st, p = scipy.stats.mannwhitneyu(auxA, auxB)
                                    # df_p_aux['test'] = 'MannWhitneyU'
                                    df_p_aux['p_value'] = [p]

                                elif test == 'RankSums':
                                    st, p = scipy.stats.ranksums(auxA, auxB)
                                    df_p_aux['test'] = 'RankSums'
                                    df_p_aux['p_value'] = [p]

                                elif test == 't-test':
                                    st, p = scipy.stats.ttest_ind(auxA, auxB)
                                    df_p_aux['test'] = 't-test'
                                    df_p_aux['p_value'] = [p]

                                df_p_aux['N'] = grid_n_l[ni]

                                # Save the name of the calculated comparison
                                df_p_aux['comparison'] = group_dict[str(c)] + '_' + group_dict[str(k)]
                                df_p_aux['test'] = 'MannWhitneyU'
                                df_p_aux['measure'] = data_features[str(m)]
                                frames = [df_pvalues, df_p_aux]
                                df_pvalues = pd.concat(frames)
                    print('Cross validation with N = ', grid_n_l[ni], ' and folds = ', folds_l[ni], ' finished.')
                
                # Store p-values after cross validation to release memory. 
                file_name = group_dict[str(c)] + '_' + group_dict[str(k)] + '_pvalues.npy'
                np.save(os.path.join(temp_folder, file_name), df_pvalues)
                file_list.append(file_name)
    # Read all saved p-values from '../computed_pvalues/'            
    df_pvalues = read_pvalues(file_list, temp_folder=temp_folder)
    return df_pvalues

def get_comparison_list(group_dict, test = None):
    # group_dict: dictionary of the different groups    
    if test is None:
        test = 'MannWhitneyU'  
    file_list=pd.DataFrame()
    # Compare each condition with the rest without repeating the comparisons        
    for c in range(len(group_dict)):
        if c+1 < len(group_dict):
            for k in range(c+1,len(group_dict)):
                print('comparison: ' + group_dict[str(c)] + '_' + group_dict[str(k)])
                                
                # Store p-values after cross validation and save some memory
                file_name = [group_dict[str(c)] + '_' + group_dict[str(k)] + '_pvalues.npy']
                file_name = pd.DataFrame(file_name)
                file_list = pd.concat([file_list,file_name])
                         
    return file_list