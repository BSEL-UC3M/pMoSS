# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 19:39:28 2018

@author: egomez
"""
import numpy as np
import scipy
import scipy.stats
import pandas as pd

def compare_continuous_samples(sample, groupA, grupB, measure_name, test = None):
    # sample: dataframe with values corresponding to group A and group B that must be compared. 
    # groupA: string. Name of the group A
    # group B: sring. Name of the group B
    # test: string. Name of the satistical test: "MannWhitneyU", "RankSums", "t-test"
    
    # Rank sum    
    if test is None:
        test = 'MannWhitneyU'
        
    df_p_aux = pd.DataFrame()    
    
    # Calculate the p-value for the required statistical test    
    if test == 'MannWhitneyU':
        st, p = scipy.stats.mannwhitneyu(sample[measure_name][groupA], sample[measure_name][grupB])
        df_p_aux['test'] = 'MannWhitneyU'
        df_p_aux['p_value'] = [p]
         
    elif test == 'RankSums':
        st, p = scipy.stats.ranksums(sample[measure_name][groupA], sample[measure_name][grupB])
        df_p_aux['test'] = 'RankSums'
        df_p_aux['p_value'] = [p]
        
    elif test == 't-test':
        st, p = scipy.stats.ttest_ind(sample[measure_name][groupA], sample[measure_name][grupB])
        df_p_aux['test'] = 't-test'
        df_p_aux['p_value'] = [p]
        
    else:
        print(test + ' not known')
        
    # Save the value in the dataframe
    
    
    return df_p_aux

def pvalues_continuous(df, group_dict, measure_name, ni, foldsi, test = None):
    # Obtain the list of p-values for a particular value ni in the grid and size foldsi. More information in the manuscript.
    
    if test is None:
        test = 'MannWhitneyU'
                                                         
    df_p = pd.DataFrame()
    
    for epoch in range(foldsi):        
        # Obtain subsamples by condition
        sample = df.groupby('Condition').apply(lambda x: x.sample(n = ni))
        
        # Compare each condition with the rest without repeating the comparisons        
        for c in range(len(group_dict)):
            if c+1 < len(group_dict):
                for k in range(c+1,len(group_dict)):     
                    
                        # Statistical test 
                        df_p_aux = compare_continuous_samples(sample, group_dict[str(c)], group_dict[str(k)], measure_name, test = test)
                        df_p_aux['N'] = ni
                        
                        # Save the name of the calculated comparison
                        df_p_aux['comparison'] = group_dict[str(c)] + '_' + group_dict[str(k)]
                        frames = [df_p, df_p_aux]
                        df_p = pd.concat(frames)
       
    return df_p

def obtain_pvalues_ChiSquared(df, group_dict, ni, foldsi):
    df_p = pd.DataFrame()
    for epoch in range(foldsi):
        
        # Calculate the cross-table with the amount of protrusion_binary = 1 or                                                   
        sample = df.groupby('Condition').apply(lambda x: x.sample(n = ni))
        cell_protrusion_tb = pd.crosstab(sample.Condition, sample.protrusion_binary, margins = True)
        observed_protrusions = cell_protrusion_tb.ix[:len(group_dict),:2]
                                                          
        aux_0 = observed_protrusions[0] == 0
        aux_0 = aux_0.astype(int)
                                                          
        aux_1 = observed_protrusions[1] == 0
        aux_1 = aux_1.astype(int)
        
        if np.sum(aux_0) == 0 and np.sum(aux_1) == 0: # Be sure that there is something to compare different from zero
            for c in range(len(group_dict)):
                if c+1 < len(group_dict):
                    for k in range(c+1,len(group_dict)):
                        # Build the table to compare one group with another.
                        df_p_aux = pd.DataFrame()
                        sampleA = 100*cell_protrusion_tb.ix[c,0:2]/cell_protrusion_tb.ix[c,2]
                        sampleB = 100*cell_protrusion_tb.ix[k,0:2]/cell_protrusion_tb.ix[k,2]
                        observed = pd.DataFrame([np.array(sampleA), np.array(sampleB)], 
                                index = [group_dict[str(c)], group_dict[str(k)]])
                        
                        # Statistical test.
                        st, p, h0, expected = scipy.stats.chi2_contingency(observed = observed)
                                                          
                        # Save the data.
                        df_p_aux['p_value'] = [p]
                        df_p_aux['N'] = ni
                        df_p_aux['comparison'] = group_dict[str(c)] + '_' + group_dict[str(k)]
                        frames = [df_p, df_p_aux]
                        df_p = pd.concat(frames)
        else: 
            epoch = epoch-1
    return df_p
