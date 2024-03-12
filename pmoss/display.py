# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 15:15:09 2019

@author: egomez
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from .models.exponential_fit import decission_data_exponential
from .models.lowess_fit import significance_analysis
from .analysis import get_decision_index
import os

def table_of_results(param, data_features, combination_dict):
    Theta = get_decision_index(param, data_features, combination_dict)

    for i in range(len(data_features)):
        t = Theta[['comparison', data_features[str(i)] + ' Theta']]
        p = param[['comparison', data_features[str(i)] + '_nalpha_estimated',
                   data_features[str(i)] + '_nalpha_theory']]
        p = p.assign(a=np.nan, c=np.nan)
        p[['a', 'c']] = param[data_features[str(i)] + '_exp_params'].apply(
            lambda x: pd.Series([x[0], x[1]], index=['a', 'c']))
        p = p[['comparison', 'a', 'c',
               data_features[str(i)] + '_nalpha_estimated',
               data_features[str(i)] + '_nalpha_theory']]
        p.rename(columns={'a': data_features[str(i)] + ' a',
                          'c': data_features[str(i)] + ' c',
                          data_features[str(i)] + '_nalpha_estimated':
                              data_features[str(i)] + ' ^n-alpha',
                          data_features[str(i)] + '_nalpha_theory':
                              data_features[str(i)] + ' n-alpha'},
                 inplace=True)

        if i == 0:
            table = pd.merge(p, t, on='comparison')
        else:
            aux = pd.merge(p, t, on='comparison')
            table = pd.merge(table, aux, on='comparison')
    return table

def func_exp_pure(x, a, c):
    return a*np.exp(-c*(x))
              
def plot_decission_with_LOWESS(df, combination,test, measure, fs = None, 
                   width = None, height = None, path = None,
                   file_name = None, colors = None, #ctheme_lowess = None, 
                   sign_level = None, gamma = None):

    """
    Function to plot the estimated p-values and the fitted exponential function
    by comparisons:
        
        - df: pandas dataframe containing the estimated p-values
        - combination: dictionary with the list of comparisons, i.e.
                        combination={
                                     '0': 'A02_A03',
                                     '1': 'A02_A09',
                                     '2': 'A02_A16',
                                     '3': 'A02_A29',
                                     '4': 'A02_A35',
                                     '5': 'A02_A55', 
                                     '6': 'A02_A65', 
                                     '7': 'A02_A85', 
                                     '8': 'A02_A96'
                                     }
        - test: dictionary,  i.e. {'0': 'MannWhitneyU'}
        - measure: measures for which the plot is done, i.e. 
                    variables={
                                '0': 'area (px^2)',
                                '1': 'short axis length (px)',
                                '2': 'orientation'
                                }
        
        Optional parameters:
            
        - ctheme: list of colors. Each measure is plot with a different color.
        - fs: optional. font size
        - height: height of the figure
        - width: width of the figure
        - path: directory in which the figure will be saved. If it is None, then 
                the image is not saved. 
        - file_name: name to use for the figure if it is saved. 
        - sign_level: alpha for statistical significance of 100(1-alpha). 
                    Set by default as 0.05.
        - gamma: threshold to calculate the convergence of p(n) and Theta function.
        
    """
      
    if colors is None:
        colors = ['#B80652', '#FF2017', '#F36635', #'#DD9952', 
                  '#CABB04', 
                  #'#A7C850', 
                  '#56C452', '#2EBAB3', '#1C69A8', '#25369A', 
                  '#4E3180']                         
    if fs is None:
        fs = 10
    if height is None:
        height = 5
            
    if width is None:
        width = 10
            
    if path is None:
        save_fig = 0
    else:
        save_fig = 1
            
    if file_name is None:
        # Change the format to png, jpg, or any other one as 
        # "file_name = 'p_values.pdf"
        file_name = 'p_values.png'
            
    if sign_level is None:
        sign_level = 0.05
            
    if gamma is None:
        gamma = 5e-06   
    param = decission_data_exponential(df, combination, measure, 
                                       sign_level = sign_level, gamma = gamma)
    N = 500  
    for c in range(len(combination)):
        param1 = param[param.comparison==combination[str(c)]]
        print(combination[str(c)])
        f = plt.gcf()
        ax = plt.gca()
        f.set_figwidth(width)
        f.set_figheight(height)
        mpl.style.use('seaborn')
        sns.set_style("white")
        sns.set_context("talk")
        splot  = ax
        labels = ['A']
        df_comparison = df[df.comparison == combination[str(c)]]
        for i in range(len(measure)):
            df_measure = df_comparison[df_comparison.measure == measure[str(i)]]
            
            for t in range(len(test)):
                # Plot LOWESS fit
                df_test = df_measure[df_measure.test == test[str(t)]]
                L, dcoeff, positive_N = significance_analysis(df_test, 
                                                    sign_level = sign_level)
                positive_N = (param1[measure[str(i)]+
                                             '_nalpha_estimated'][0])
                splot.plot(L[:,0], L[:,1], color=colors[i])# color = ctheme[i],
#                splot.fill_between(L[:,0], L[:,1], 0*L[:,1], color = ctheme_lowess[i], alpha=0.7)
                labels = np.concatenate((labels, [measure[str(i)] +
                                                r' $\hat{n}_\alpha$ = ' +
                                                str(positive_N)]))
    
                # EXPONENTIAL FIT
                par_a,par_c= (param1[measure[str(i)]+'_exp_params'][0])
                positive_N = (param1[measure[str(i)]+
                                             '_nalpha_theory'][0])
                splot.plot(np.arange(0,N), 
                             func_exp_pure(np.arange(0,N),par_a,par_c),
                             linestyle='--', color=colors[i])# color = ctheme[i],

                labels = np.concatenate((labels, [r'Exponential fit $n_{\alpha}$ = ' + 
                                        str(positive_N)]))
            
            splot.tick_params(labelsize = fs)
        y = sign_level*np.ones((len(np.arange(0,N))))
        splot.plot(np.arange(0,N), y, color = 'black')
        labels = np.concatenate((labels, [r'$\alpha = 0.05$']))
        splot.legend(labels[1:], bbox_to_anchor=(1, 1),ncol = 1,fancybox=True,
                     shadow=True, fontsize = fs) # loc='best', 
        
        splot.set_title(combination[str(c)], fontsize = fs)
        splot.set_xlabel('Sample size (n)', fontsize = fs)
        splot.set_ylabel('p-value ' + combination[str(c)], fontsize = fs)
        splot.set_ylim([0,0.45])
        f.tight_layout()
        if save_fig == 1:
            plt.savefig(os.path.join(path,  combination[str(c)] + file_name), dpi=75)
        plt.show()
   
    
def plot_pcurve_by_measure(df, combination, measure, test = None,  fs = None, 
                   width = None, height = None, path = None,
                   file_name = None, colors = None, #ctheme_lowess = None, 
                   sign_level = None, gamma = None):
    """
    Function to plot the estimated p-values and the fitted exponential function
    by measures:
        
        - df: pandas dataframe containing the estimated p-values
        - combination: dictionary with the list of comparisons, i.e.
                        combination={
                                     '0': 'A02_A03',
                                     '1': 'A02_A09',
                                     '2': 'A02_A16',
                                     '3': 'A02_A29',
                                     '4': 'A02_A35',
                                     '5': 'A02_A55', 
                                     '6': 'A02_A65', 
                                     '7': 'A02_A85', 
                                     '8': 'A02_A96'
                                     }
        
        - measure: measures for which the plot is done, i.e. 
                    variables={
                                '0': 'area (px^2)',
                                '1': 'short axis length (px)',
                                '2': 'orientation'
                                }
        
        Optional parameters:
        - test: dictionary,  i.e. {'0': 'MannWhitneyU'}
        - ctheme: list of colors. Each measure is plot with a different color.
        - fs: optional. font size
        - height: height of the figure
        - width: width of the figure
        - path: directory in which the figure will be saved. If it is None, then 
                the image is not saved. 
        - file_name: name to use for the figure if it is saved. 
        - sign_level: alpha for statistical significance of 100(1-alpha). 
                    Set by default as 0.05.
        - gamma: threshold to calculate the convergence of p(n) and Theta function.
        
    """
    
    if colors is None:
        colors = ['#B80652', '#FF2017', '#F36635', #'#DD9952', 
                  '#CABB04', 
                  #'#A7C850', 
                  '#56C452', '#2EBAB3', '#1C69A8', '#25369A', 
                  '#4E3180']                        
    if fs is None:
        fs = 10
    if height is None:
        height = 8
            
    if width is None:
        width = 10
            
    if path is None:
        save_fig = 0
    else:
        save_fig = 1
            
    if file_name is None:
        # Change the format to png, jpg, or any other one as 
        # "file_name = 'p_values.pdf"
        file_name = 'p_values.png'
            
    if sign_level is None:
        sign_level = 0.05
    
    if test is None:
        test ={'0': 'MannWhitneyU'}
           
    if gamma is None:
        gamma = 5e-06  
    param = decission_data_exponential(df, combination, measure, 
                                       sign_level = sign_level, gamma = gamma)
    N = max(df.N)
    # N = 1200 # 400  
    
    for i in range(len(measure)):
        df_measure = df[df.measure == measure[str(i)]]
        print(measure[str(i)])
        y_max = 0
        f = plt.gcf()
        ax = plt.gca()
        f.set_figwidth(width)
        f.set_figheight(height)
        mpl.style.use('seaborn')
        sns.set_style("white")
        sns.set_context("talk")
        splot  = ax
        labels = ['A']        
        for c in range(len(combination)):
            param1 = param[param.comparison==combination[str(c)]]
            df_comparison = df_measure[df_measure.comparison == combination[str(c)]]
            
            for t in range(len(test)):
                
                # Plot LOWESS fit
                df_test = df_comparison[df_comparison.test == test[str(t)]]
                
                L, dcoeff, positive_N = significance_analysis(df_test,
                                                    sign_level = sign_level)
                positive_N = (param1[measure[str(i)]+
                                    '_nalpha_estimated'][0])
                splot.plot(L[:,0], L[:,1], color=colors[c])
                y_max = max(y_max,max(L[:,1]))
                labels = np.concatenate((labels, [combination[str(c)] +
                                                r' $\hat{n}_\alpha$ = ' +
                                                str(positive_N)]))
                # EXPONENTIAL FIT
                par_a,par_c= (param1[measure[str(i)]+'_exp_params'][0])
                positive_N = (param1[measure[str(i)]+
                                    '_nalpha_theory'][0])                
                splot.plot(np.arange(0,N), 
                             func_exp_pure(np.arange(0,N),par_a,par_c),
                             linestyle='--', color=colors[c])
                labels = np.concatenate((labels,
                                         [r'Exponential fit $n_{\alpha}$ = ' +
                                          str(positive_N)]))
            
            splot.tick_params(labelsize = fs)
        y = sign_level*np.ones((len(np.arange(0,N))))
        splot.plot(np.arange(0,N), y, color = 'black')
        labels = np.concatenate((labels, [r'$\alpha = 0.05$']))
        splot.legend(labels[1:], bbox_to_anchor=(1, 1),ncol = 1,fancybox=True,
                     fontsize = fs) # loc='best', 
        splot.set_title(measure[str(i)], fontsize = fs)
        splot.set_xlabel('Sample size (n)', fontsize = fs)
        splot.set_ylabel('p-value ' + measure[str(i)], fontsize = fs)
#        splot.set_ylim([0,0.45])
        splot.set_ylim([0,y_max])
        
        f.tight_layout()
        if save_fig == 1:
            plt.savefig(os.path.join(path, measure[str(i)] + file_name), dpi=75)
        plt.show()


def scatterplot_decrease_parameters(df, combination,measure,plot_type="exp-param",
                                    fs = None, width = None, height = None, 
                                    path = None, file_name = None, 
                                    colors = None, #ctheme_lowess = None, 
                                    sign_level = None, gamma = None):
    """
    Function to plot the exponential parameters or the estimated and 
    theoretical minimum size n-alpha .
        
        - df: pandas dataframe containing the estimated p-values
        - combination: dictionary with the list of comparisons, i.e.
                        combination={
                                     '0': 'A02_A03',
                                     '1': 'A02_A09',
                                     '2': 'A02_A16',
                                     '3': 'A02_A29',
                                     '4': 'A02_A35',
                                     '5': 'A02_A55', 
                                     '6': 'A02_A65', 
                                     '7': 'A02_A85', 
                                     '8': 'A02_A96'
                                     }
        - measure: measures for which the plot is done, i.e. 
                    variables={
                                '0': 'area (px^2)',
                                '1': 'short axis length (px)',
                                '2': 'orientation'
                                }
        - plot_type="exp-param","sampled-nalpha" or "theory-nalpha", for 
                    exponential parameters a and c, estimated minimum data size
                    or theoretical minimum data size respectively.         
        Optional parameters:
            
        - ctheme: list of colors. Each measure is plot with a different color.
        - fs: optional. font size
        - height: height of the figure
        - width: width of the figure
        - path: directory in which the figure will be saved. If it is None, then 
                the image is not saved. 
        - file_name: name to use for the figure if it is saved. 
                    Specify the file format as well:  file_name.png
        - sign_level: alpha for statistical significance of 100(1-alpha). 
                    Set by default as 0.05.
        - gamma: threshold to calculate the convergence of p(n) and Theta function.
        
    """  
    if colors is None:
        colors = ['#B80652', '#FF2017', '#F36635', #'#DD9952', 
                  '#CABB04', 
                  #'#A7C850', 
                  '#56C452', '#2EBAB3', '#1C69A8', '#25369A', 
                  '#4E3180']
    if fs is None:
        fs = 10
    if height is None:
        height = 8
            
    if width is None:
        width = 10
            
    if path is None:
        save_fig = 0
    else:
        save_fig = 1
            
    if file_name is None:
        if plot_type == "exp-param":
            file_name = 'scatter_exp_params.png'
            
        elif plot_type == "sampled-nalpha":
            file_name = 'scatter_sampled_minimum_nalpha.pdf'
            
        elif plot_type == "theory-nalpha":
            file_name = 'scatter_theory_minimum_nalpha.pdf'
            
    if sign_level is None:
        sign_level = 0.05
            
    if gamma is None:
        gamma = 5e-06  
    param = decission_data_exponential(df, combination, measure, 
                                       sign_level = sign_level, gamma = gamma)
    markers = ['o', '^', 's', '.','<','<', '>', 's', 'd']
    for i in range(len(measure)):
        print(measure[str(i)])
        f = plt.gcf()
        ax = plt.gca()
        f.set_figwidth(width)
        f.set_figheight(height)
        mpl.style.use('seaborn')
        sns.set_style("white")
        sns.set_context("talk")
        splot  = ax
        labels = ['A']
        # initialize values for the plot axis
        ma = 1.0
        Ma = 0.0
        Mc = 0.0
        MN = 0.0
        for c in range(len(combination)):
            param1 = param[param.comparison==combination[str(c)]]
            
            if plot_type == "exp-param": 
                # plot exponential parameters
                par_a,par_c= (param1[measure[str(i)]+'_exp_params'][0])
                splot.plot(par_c,par_a,markers[0], color=colors[c])
                ma = np.min((ma,par_a))
                Ma = np.max((Ma,par_a))         
                Mc = np.max((Mc,par_c))
                labels = np.concatenate((labels, [combination[str(c)]]))
                
            elif plot_type == "sampled-nalpha":
                nalpha = (param1[measure[str(i)]+'_nalpha_estimated'][0])
                if not np.isnan(nalpha):
                    # plot n alpha values
                    splot.plot(nalpha,markers[0], color=colors[c])
                    MN = np.max((MN,nalpha))  
                    labels = np.concatenate((labels, [combination[str(c)]]))
                    
            elif plot_type == "theory-nalpha":
                nalpha = (param1[measure[str(i)]+'_nalpha_theory'][0])
                if not np.isnan(nalpha):
                    # plot n alpha values
                    splot.plot(nalpha,markers[0], color=colors[c])
                    MN = np.max((MN,nalpha))  
                    labels = np.concatenate((labels, [combination[str(c)]]))

        if plot_type == "exp-param":            
            splot.set_ylim([ma-0.01, Ma+0.01])
            splot.set_xlim([-0.01, Mc+0.05])
            splot.set_xlabel('c', fontsize = fs)
            splot.set_ylabel('a', fontsize = fs)
            
        elif plot_type == "sampled-nalpha" or plot_type == "theory-nalpha":
            splot.set_ylim([-0.01,MN+50])
#            splot.set_xlabel('None', fontsize = fs)
            splot.set_ylabel(r'$n_\alpha$', fontsize = fs)
            splot.set_xlim([-0.01,0.01])
            splot.xaxis.set_ticklabels([])
#        plt.axes(xscale='log', yscale='log')
        if plot_type == "exp-param":
            # plot a line in c=0 to indicate those uniform distributions            
            splot.plot([0,0], [ma-0.01, Ma+0.01], color = 'grey')     
        splot.legend(labels[1:], bbox_to_anchor=(1, 1),ncol = 1,fancybox=True, 
                     shadow=True, fontsize = fs) # loc='best', 
        splot.set_title(measure[str(i)], fontsize = fs)
        splot.tick_params(labelsize = fs,length = 5,colors='black')
        f.tight_layout()
        if save_fig == 1:
            plt.savefig(path + measure[str(i)] + file_name, dpi=75)
        plt.show()
        
#def scatterplot_parameters_theta(df, combination,measure,fs = None, 
#                   width = None, height = None, path = None,
#                   file_name = None, ctheme = None, #ctheme_lowess = None, 
#                   sign_level = None, gamma = None):
#    """
#    Function to plot the mean value of normal distributions being compared 
#    versus the estimated exponential parameter c, and indicating whether the 
#    decision index Theta is 0 or 1.
#        
#        - df: pandas dataframe containing the estimated p-values
#        - combination: dictionary with the list of comparisons, i.e.
#                        combination={
#                                     '0': 'A02_A03',
#                                     '1': 'A02_A09',
#                                     '2': 'A02_A16',
#                                     '3': 'A02_A29',
#                                     '4': 'A02_A35',
#                                     '5': 'A02_A55', 
#                                     '6': 'A02_A65', 
#                                     '7': 'A02_A85', 
#                                     '8': 'A02_A96'
#                                     }
#        - measure: measures for which the plot is done, i.e. 
#                    variables={
#                                '0': 'area (px^2)',
#                                '1': 'short axis length (px)',
#                                '2': 'orientation'
#                                }
#        - plot_type="exp-param","sampled-nalpha" or "theory-nalpha", for 
#                    exponential parameters a and c, estimated minimum data size
#                    or theoretical minimum data size respectively.         
#        Optional parameters:
#            
#        - ctheme: list of colors. Each measure is plot with a different color.
#        - fs: optional. font size
#        - height: height of the figure
#        - width: width of the figure
#        - path: directory in which the figure will be saved. If it is None, then 
#                the image is not saved. 
#        - file_name: name to use for the figure if it is saved. 
#        - sign_level: alpha for statistical significance of 100(1-alpha). 
#                    Set by default as 0.05.
#        - gamma: threshold to calculate the convergence of p(n) and Theta function.
#        
#    """  
#    if ctheme is None:
#        colors = ['#B80652', '#FF2017', '#F36635', #'#DD9952', 
#                  '#CABB04', 
#                  #'#A7C850', 
#                  '#56C452', '#2EBAB3', '#1C69A8', '#25369A', 
#                  '#4E3180']                    
#    if fs is None:
#        fs = 10
#    if height is None:
#        height = 5
#            
#    if width is None:
#        width = 4
#            
#    if path is None:
#        save_fig = 0
#    else:
#        save_fig = 1
#            
#    if file_name is None:
#        file_name = 'scatter_parameters_theta.png'
#            
#    if sign_level is None:
#        sign_level = 0.05
#            
#    if gamma is None:
#        gamma = 5e-06 
#    param = decission_data_exponential(df, combination, measure, 
#                                       sign_level = sign_level, gamma = gamma)
#    markers = ['o', '^', 's', '.','<','<', '>', 's', 'd']
#    for i in range(len(measure)):
#        print(measure[str(i)])
#        f = plt.gcf()
#        ax = plt.gca()
#        f.set_figwidth(width)
#        f.set_figheight(height)
#        mpl.style.use('seaborn')
#        sns.set_style("white")
#        sns.set_context("talk")
#        splot  = ax
#        labels = ['A']
#        for c in range(len(combination)):
#            param1 = param[param.comparison==combination[str(c)]]
#            par_a,par_c= (param1[measure[str(i)]+'_exp_params'][0])
#            dist = (param1[measure[str(i)] + '_convergence_d'][0])
#            mean_value = combination[str(c)]
#            mean_value = float(mean_value[18:mean_value.find('_',19)])
#            
#            if dist == np.inf:
#                 markers = 's'
#                 dist = 0
#            elif dist>=0:
#                markers = 's'
#            else:
#                markers = 'o'
#
#            splot.plot(mean_value,par_c,markers, color=colors[c])
#            labels = np.concatenate((labels, [combination[str(c)]]))
#
#            splot.tick_params(labelsize = fs)
#        splot.legend(labels[1:], bbox_to_anchor=(1, 1),ncol = 1,fancybox=True, 
#                     shadow=True, fontsize = fs) # loc='best', 
#        splot.set_title(measure[str(i)], fontsize = fs)
#        splot.set_xlabel('mean value (\mu)', fontsize = fs)
#        splot.set_ylabel('c', fontsize = fs)
##        splot.set_ylim([0,0.4])
#        splot.set_xlim([-0.1,3.1])
#
#        if save_fig == 1:
#            plt.savefig(path + measure[str(i)] + file_name, dpi=75)
#        plt.show()    
        
def composed_plot(data, df, group_labels, combination,measure,test = None,
                                    fs = None, width = None, height = None, 
                                    path = None, file_name = None, 
                                    colors = None, #ctheme_lowess = None, 
                                    sign_level = None, gamma = None,
                                    bins = None):
    """
    Function to plot the exponential parameters or the estimated and 
    theoretical minimum size n-alpha .
        
        - df: pandas dataframe containing the estimated p-values
        - combination: dictionary with the list of comparisons, i.e.
                        combination={
                                     '0': 'A02_A03',
                                     '1': 'A02_A09',
                                     '2': 'A02_A16',
                                     '3': 'A02_A29',
                                     '4': 'A02_A35',
                                     '5': 'A02_A55', 
                                     '6': 'A02_A65', 
                                     '7': 'A02_A85', 
                                     '8': 'A02_A96'
                                     }
        - measure: measures for which the plot is done, i.e. 
                    variables={
                                '0': 'area (px^2)',
                                '1': 'short axis length (px)',
                                '2': 'orientation'
                                }
        - plot_type="exp-param","sampled-nalpha" or "theory-nalpha", for 
                    exponential parameters a and c, estimated minimum data size
                    or theoretical minimum data size respectively.         
        Optional parameters:
            
        - ctheme: list of colors. Each measure is plot with a different color.
        - fs: optional. font size
        - height: height of the figure
        - width: width of the figure
        - path: directory in which the figure will be saved. If it is None, then 
                the image is not saved. 
        - file_name: name to use for the figure if it is saved. 
                    Specify the file format as well:  file_name.png
        - sign_level: alpha for statistical significance of 100(1-alpha). 
                    Set by default as 0.05.
        - gamma: threshold to calculate the convergence of p(n) and Theta function.
        
    """  
    if colors is None:
        colors = ['#B80652', '#FF2017', '#F36635', #'#DD9952', 
                  '#CABB04', 
                      '#A7C850', 
                  '#56C452', '#2EBAB3', '#1C69A8', '#25369A', 
                  '#4E3180']
    if fs is None:
        fs = 10
    if height is None:
        height = 6
            
    if width is None:
        width = 15
            
    if path is None:
        save_fig = 0
    else:
        save_fig = 1
            
    if file_name is None:
        file_name = 'composed_figure.pdf'
            
    if sign_level is None:
        sign_level = 0.05
        
    if test is None:
        test ={'0': 'MannWhitneyU'}
            
    if gamma is None:
        gamma = 5e-06  
    if bins is None:
        bins = 200
    param = decission_data_exponential(df, combination, measure, 
                                       sign_level = sign_level, gamma = gamma)
    markers = ['o', '^', 's', '.','<','<', '>', 's', 'd']
    N = max(df.N)
    for i in range(len(measure)):
        print(measure[str(i)])
        df_measure = df[df.measure == measure[str(i)]]
        f = plt.gcf()        
        f.set_figwidth(width)
        f.set_figheight(height)
        mpl.style.use('seaborn')
        sns.set_style("white")
        sns.set_context("talk")
        
        for gl in range(len(group_labels)):
            if gl == 0:
                aux = [data[data['Condition'] == group_labels[str(gl)]][measure[str(i)]]]
            else:
                aux = aux + [data[data['Condition'] == group_labels[str(gl)]][measure[str(i)]]]
        ## 
        splot = plt.subplot(141)        
        kwargs = dict(histtype='step', alpha = 0.6, bins = bins, density = False,
                      cumulative = False, linewidth = 2)
        n, bins_hist, patches = plt.hist(aux, label = list(group_labels.values()), 
                                    color = colors[:len(group_labels)], **kwargs)
        splot.legend(fontsize = fs)
        splot.set_xlim([np.min(data[measure[str(i)]]),np.max(data[measure[str(i)]])])
        splot.tick_params(labelsize = fs, length = 5, colors='black')
        splot.set_title(measure[str(i)], fontsize = fs)
        
#        plt.show()

        labels = ['A']
        # initialize values for the plot axis
        ma = 1.0
        Ma = 0.0
        Mc = 0.0
        MN = 0.0
        y_max = 0
        par_a = np.zeros(len(combination))
        par_c = np.zeros(len(combination))
        nalpha = np.zeros(len(combination))
        nalpha_th = np.zeros(len(combination))
        
        for c in range(len(combination)):
            param1 = param[param.comparison==combination[str(c)]]
            df_comparison = df_measure[df_measure.comparison == combination[str(c)]]
            # plot exponential parameters
            par_a[c],par_c[c]= (param1[measure[str(i)]+'_exp_params'][0])
            nalpha[c] = (param1[measure[str(i)]+'_nalpha_estimated'][0])
            nalpha_th[c] = (param1[measure[str(i)]+ '_nalpha_theory'][0])
            
            ma = np.min((ma,par_a[c]))
            Ma = np.max((Ma,par_a[c]))         
            Mc = np.max((Mc,par_c[c]))
            
            if not np.isnan(nalpha[c]):
                splot = plt.subplot(142)
                splot.plot(nalpha[c], markers[0], color=colors[c], 
                           markersize = 20)
                MN = np.max((MN,nalpha[c]))
            
            splot = plt.subplot(143)
            splot.plot(par_c[c],par_a[c],markers[0], color=colors[c],
                       markersize = 20)
            
#            labels = np.concatenate((labels, [combination[str(c)]]))
            for t in range(len(test)):
                # LOWESS fit
                df_test = df_comparison[df_comparison.test == test[str(t)]]
                L, dcoeff, positive_N = significance_analysis(df_test,
                                                    sign_level = sign_level)                
                y_max = max(y_max,max(L[:,1]))                
                
                ##
                splot = plt.subplot(144)
                splot.plot(L[:,0], L[:,1], color=colors[c])
                splot.plot(np.arange(0,N), 
                             func_exp_pure(np.arange(0,N),par_a[c],par_c[c]),
                             linestyle='--', color=colors[c])
                labels = np.concatenate((labels, [combination[str(c)] +
                                                r' $\hat{n}_\alpha$ = ' +
                                                str(nalpha[c])]))
                labels = np.concatenate((labels,
                                         [r'Exponential fit $n_{\alpha}$ = ' +
                                          str(nalpha_th[c])]))
        ##
        splot = plt.subplot(142)
        splot.set_ylim([-0.01,MN+50])
        splot.set_ylabel(r'$n_\alpha$', fontsize = fs)
        splot.set_xlim([-0.01,0.01])
        splot.set_title(measure[str(i)], fontsize = fs)
        splot.xaxis.set_ticklabels([])
        splot.tick_params(labelsize = fs,length = 5,colors='black')
        
        ##
        splot = plt.subplot(143)     
        splot.plot([0,0], [ma-0.01, Ma+0.01], color = 'grey')
        splot.set_ylim([ma-0.01, Ma+0.01])
        splot.set_xlim([-0.01, Mc+0.05])
        splot.set_xlabel('c', fontsize = fs)
        splot.set_ylabel('a', fontsize = fs)
        splot.set_title(measure[str(i)], fontsize = fs)
        splot.tick_params(labelsize = fs,length = 2,colors='black')
        
        ##
        splot = plt.subplot(144) 
        y = sign_level*np.ones((len(np.arange(0,N))))
        splot.plot(np.arange(0,N), y, color = 'black')
        labels = np.concatenate((labels, [r'$\alpha = 0.05$']))
        splot.legend(labels[1:], bbox_to_anchor=(1, 1),ncol = 1,fancybox=True, 
                     shadow=True, fontsize = fs)
        splot.set_title(measure[str(i)], fontsize = fs)
        splot.tick_params(labelsize = fs,length = 5,colors='black')
        splot.set_xlabel('Sample size (n)', fontsize = fs)
        splot.set_ylabel('p-value ' + measure[str(i)], fontsize = fs)
        splot.set_ylim([0,y_max])
        
#        f.tight_layout()
        if save_fig == 1:
            plt.savefig(os.path.join(path, measure[str(i)] + file_name), dpi=75)
        plt.show()
