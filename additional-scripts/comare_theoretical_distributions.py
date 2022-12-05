# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:59:34 2019

@author: egomez
"""
import scipy.stats
import sys
import numpy as np
import pandas as pd
import seaborn as sns # version 0.9.0 recommended
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.insert(0, './functions4diagnosis_v1')
#from exponential_fit import decission_data_exponential
#from decision_index import get_decision_index
from robustness import get_grids
from preprocessing_functions import determination_coeff
from preprocessing_functions import convergence_point


# Function for theoretical normal distributions
def compare_dist(par1, par2, n0=None, ninf=None, Nmax=None, nsize=None, k=None, initial_portion=None):
    # Cross validation parameters
    if n0 is None:
        n0=2
    if ninf is None:
        ninf=2500
    if Nmax is None:
        Nmax=10000          # number of datapoints available
    
    if nsize is None:
        nsize = 250             # grid size
    if k is None:
        k = 20
    if initial_portion is None:
        initial_portion = 1/3.
    
    # Grid definition
    grid_n, folds = get_grids(n0, ninf, Nmax, nsize = nsize, k = k, initial_portion = initial_portion)
    data_cv = pd.DataFrame()
    for i in range(len(grid_n)):        
        for f in range(folds[i]):
            sample1 = np.random.normal(loc=par1["mean"], scale=par1["std"], size=grid_n[i])
            sample2 = np.random.normal(loc=par2["mean"], scale=par2["std"], size=grid_n[i])
            pvalue = scipy.stats.mannwhitneyu(sample1,sample2)[1]
            
            sampl1_name = 'normal_'+np.str(par1["mean"])+'_'+np.str(par1["std"])
            sampl2_name = 'normal_'+np.str(par2["mean"])+'_'+np.str(par2["std"])
            df=pd.DataFrame()
            df['N'] = [np.int(grid_n[i])]
            df['p_value'] = np.float64(pvalue)
            df['comparison'] = sampl1_name + '_' + sampl2_name
            df['test'] = 'MannWhitneyU'
            df['measure'] = "normal"
            data_cv=pd.concat([data_cv,df])
    return data_cv
#par1 = {
#  "dist": "Normal",
#  "mean": 0,
#  "std": 1
#}
#par2 = {
#  "dist": "Normal",
#  "mean": 0.5,
#  "std": 1
#}
#
#
#data_cv=pd.DataFrame()
#mean_values = [0,0.01,0.1,0.25,0.5,0.75,1,1.5,2,2.5,3]
#for i in mean_values:
#    par2['mean'] = i
#    aux = compare_dist(par1, par2)
#    data_cv=pd.concat([data_cv,aux])
#np.save('C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/DISCRIMINATIVE_ANALYSIS/data/normal_cross_validation.npy',data_cv)
#import researchpy as rp    
### Summary of data
#rp.summary_cont(data_cv['p_value'].groupby(data_cv['comparison']))
#data_cv = pd.DataFrame()
#aux = np.load('C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/DISCRIMINATIVE_ANALYSIS/data/normal_cross_validation.npy')
#data_cv['N'] = aux[:,0].astype(np.int32)
#data_cv['p_value'] = aux[:,1].astype(np.float32)
#data_cv['comparison'] = aux[:,2]
#data_cv['test'] = aux[:,3]
#data_cv['measure'] = aux[:,4]
#del(aux) 
#measure={'0': 'normal'}    
#comparison={'0': 'normal_0_1_normal_0_1',
#            '1': 'normal_0_1_normal_0.01_1',
#            '2': 'normal_0_1_normal_0.1_1',
#            '3': 'normal_0_1_normal_0.25_1',
#            '4': 'normal_0_1_normal_0.5_1',
#            '5': 'normal_0_1_normal_0.75_1',
#            '6': 'normal_0_1_normal_1_1',
#            '7': 'normal_0_1_normal_1.5_1',
#            '8': 'normal_0_1_normal_2_1',
#            '9': 'normal_0_1_normal_2.5_1',
#            '10': 'normal_0_1_normal_3_1',
#        }
#mean_values = [0,0.01,0.1,0.25,0.5,0.75,1,1.5,2,2.5,3]    
#
################################################################################
## Plot with boxplots
################################################################################
#
#for c in range(len(comparison)):
#    pdata = data_cv[data_cv['comparison'] == comparison[np.str(c)]]
##
##    f = plt.gcf()
##    ax = plt.gca()
##    f.set_figwidth(15)
##    f.set_figheight(5)
##    sns.set(font_scale = 1)
##    sns.set_style("white")
##    meanprops={"markerfacecolor":"red", "markeredgecolor":"red"}
##    ax = sns.boxplot(x = 'N', y = 'p_value', data = pdata,notch = False,
##                     width = 0.6, whis=0.5, saturation = 100,showmeans=True,
##                     meanprops=meanprops,
##                     palette = reversed(sns.color_palette("BuPu")))
##    ax = sns.barplot(x = 'N', y = 'p_value', data = pdata,saturation = 100,
                     #showmeans=True,
##                     meanprops=meanprops,
##                     palette = reversed(sns.color_palette("BuPu")))

##    sns.set_palette("winter",5)
##    #ax.legend(labels)
##    plt.title('Normal(0,1) vs. Normal('+ np.str(mean_values[c]) + ',1)')
##    plt.show()    
    pdata_n = pd.DataFrame()
#    num = [3, 5,10,15,30,45,60,90,200, 670,1485,2499]
    num = grid_n[np.concatenate((np.array([1,50,100,112,120,125,129,135,148,187,178,179,180,181,183]),np.arange(140,len(grid_n),2)))]
    for i in range(len(num)):
        # pdata_n = pdata[pdata['N'] == n_dist[i]]
        aux = pdata[pdata['N'] == np.int(num[i])]
        pdata_n = pd.concat([pdata_n,aux])
#        print(np.mean(pdata_n['p_value'].astype(np.float)))
    pdata_n['p_value'] = pdata_n['p_value'].astype(np.float)
    f = plt.gcf()
    ax = plt.gca()
    f.set_figwidth(5)
    f.set_figheight(5)
    meanprops={"markerfacecolor":"red", "markeredgecolor":"red"}
#    ax = sns.boxplot(x = 'N', y = 'p_value', data = pdata_n,notch = False,
#                     width = 0.6, whis=0.5, saturation = 100, showmeans=True,
#                        order=num, meanprops = meanprops,
##                        color =colors[c])
#                        palette = reversed(sns.color_palette("YlGnBu",len(num)+1)))
#                        palette = reversed(sns.color_palette("rocket",len(num)+10)))
    ax = sns.barplot(x = 'N', y = 'p_value', data = pdata_n,ci=95,
                    palette = reversed(sns.color_palette("rocket",len(num)+10)))
#    sns.set_palette("winter",5)
    labels = num
    plt.title('Normal(0,1) vs. Normal('+ np.str(mean_values[c]) + ',1)')
    plt.xlabel('Sample size (n)')
    plt.ylabel('p-value')
    sns.set(font_scale = 1)
    sns.set_style("white")
    plt.show()
    f.savefig('C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/DISCRIMINATIVE_ANALYSIS/FIGURES/'+
               comparison[np.str(c)]+'.pdf', format='pdf', dpi = 350)
###
###############################################################################
# Plot convergence parameters
###############################################################################
import sys
#sys.path.append('C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/DISCRIMINATIVE_ANALYSIS/data_aging/DISCRIMINATING/functions4diagnosis_v1/')
sys.path.append('C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/DISCRIMINATIVE_ANALYSIS/public_code/DISpANALYSIS/DISpANALYSIS/')
from graphics import scatterplot_decrease_parameters
#
scatterplot_decrease_parameters(data_cv, comparison,measure,exp_params = False,
                                path ='C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/DISCRIMINATIVE_ANALYSIS/FIGURES/')
#data_cv = data_cv[data_cv.N < 200]
from graphics import plot_decission_with_LOWESS, plot_pcurve_by_measure, scatterplot_parameters_theta
test_continuous = {'0': 'MannWhitneyU'}

from exponential_fit import decission_data_exponential
param = decission_data_exponential(data_cv, comparison, measure, sign_level = 0.05, gamma = 0.001 )

from table_of_results import table_of_results
# print the results:
table = table_of_results(param, measure, comparison)
table


#plot_pcurve_by_measure(data_cv, comparison,test_continuous, measure,
#                       path ='C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/DISCRIMINATIVE_ANALYSIS/FIGURES/all_normals_')
#plot_decission_with_LOWESS(data_cv, ,test_continuous, measure)
#                           path ='C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/DISCRIMINATIVE_ANALYSIS/FIGURES/')
#scatterplot_parameters_theta(data_cv, comparison,measure
#                             path ='C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/DISCRIMINATIVE_ANALYSIS/FIGURES/all_normals_')
######

################################################################################
## Plot decision curves and exponential fitting
################################################################################
#    
## function defined at the end   
#sign_level = 0.05
##df, measure, test = load_pvalue_data('cell morphology')
#
#measure={'0': 'normal'}    
##comparison={'0': 'normal_0_1_normal_0_1',
##            '1': 'normal_0_1_normal_0.01_1',
##            '2': 'normal_0_1_normal_0.1_1',
##            '3': 'normal_0_1_normal_0.25_1',
##            '4': 'normal_0_1_normal_0.5_1',
###            '5': 'normal_0_1_normal_0.75_1',
###            '6': 'normal_0_1_normal_1_1',
###            '7': 'normal_0_1_normal_2_1',
###            '8': 'normal_0_1_normal_3_1',
##        }
##mean_values = [0,0.01,0.1,0.25,0.5,0.75,1,2,3] 
#
#mean_values = [0.75,1,2,3]     
#comparison={'0':  'normal_0_1_normal_0.75_1',
#            '1': 'normal_0_1_normal_1_1',
#            '2': 'normal_0_1_normal_2_1',
#            '3': 'normal_0_1_normal_3_1',
#        }
#   
#test_continuous = {  '0': 'MannWhitneyU' }
#data_cv['test'] = test_continuous['0']
#gamma = 0.001     
#epsilon = gamma*sign_level
#
##path = 'Z:/3D_PROTUCEL/unet_results/ALEXANDRA_Results/kfolds_statisticalanalysis/HD/'
#path = 'C:/Users/egomez/Documents/Projectos/3D-PROTUCEL/DISCRIMINATIVE_ANALYSIS/'
#file_name = 'cocient.svg'
#width = 10
#fs = 15
#height = 7
#ctheme = ['seagreen','olive', 'cornflowerblue','fuchsia','darkturquoise', 'blue','skyblue', 'cornflowerblue','seagreen','fuchsia']
##ctheme = ['darkturquoise', 'blue','skyblue', 'cornflowerblue', 'seagreen','slategray','olive', 'darkkhaki','orange','fuchsia']
#data_cv_1 = data_cv[data_cv['N']<3000]
#plot_decission_groups(data_cv_1, comparison,mean_values,test_continuous, measure, fs = fs, width = width, 
#               height = height, path = path, file_name = file_name,
#               ctheme = ctheme, sign_level = sign_level, epsilon = epsilon, ylim = 0.30)    
#    
################################################################################   
### Functions for plots    
#    
#def func_exp_pure(x, a, c):    
#    return a*np.exp(-c*(x))
#
#def plot_decission_groups(df, combination,mean_values,test, measure, fs = None, 
#                   width = None, height = None, path = None,
#                   file_name = None, ctheme = None, ctheme_lowess = None, 
#                   sign_level = None, epsilon = None, ylim = None):
#    if ctheme is None:
#        ctheme = ['teal', '#1e488f', 'r', 'tan', 'gray', '#be03fd','fuchsia']
#    if ctheme_lowess is None:
#        ctheme_lowess = ['teal', '#1e488f', 'r', 'tan', 'gray', '#be03fd']
#    if fs is None:
#        fs = 10            
#    if height is None:
#        height = 10
#    if width is None:
#        width = 10
#    if path is None:
#        save_fig = 0
#    else:
#        save_fig = 1
#    if file_name is None:
#        file_name = 'p_values.png'
#    if sign_level is None:
#        sign_level = 0.05
#    if epsilon is None:
#        epsilon = 0.0001
#    if ylim is None:
#        ylim = 1
#    f = plt.gcf()
#    ax = plt.gca()
#    f.set_figwidth(width)
#    f.set_figheight(height)  
#    labels = ['A']
#    for c in range(len(combination)):
##        f = plt.gcf()
##        ax = plt.gca()
##        f.set_figwidth(width)
##        f.set_figheight(height)
#        df_comparison = df[df.comparison == combination[np.str(c)]]
#        name = 'Normal(0,1) vs. Normal('+ np.str(mean_values[c]) + ',1)'
#        mpl.style.use('seaborn')
#        sns.set_style("white")
#        sns.set_context("talk")
#        splot  = ax
##        labels = ['A']                 
#        for i in range(len(measure)):            
#            df_measure = df_comparison[df_comparison.measure == measure[np.str(i)]]            
#            for t in range(len(test)):
#                df_test = df_measure[df_measure.test == test[np.str(t)]]
#                L, dcoeff, positive_N = determination_coeff(df_test, sign_level = sign_level)
#               ## Obtain the derivative
#                x = L[:,0]
#                y = L[:,1]
#                dy = np.zeros(y.shape,np.float)
#                dy[0:-1] = np.diff(y)/np.diff(x)
#                dy[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
#                cocient = np.divide(dy,L[:,1])
#                cocient[np.isnan(cocient)]=0
#                splot.plot(L[:,0],cocient,color = ctheme[c], alpha=0.7,linestyle='--')
#                labels = np.concatenate((labels, [name + ' cocient']))
#                ## Exponential fitting
##                splot.plot(L[:,0], L[:,1], color = ctheme[c])
##                labels = np.concatenate((labels,[name + ' LOWESS (p(n))'])) # ' d = ' + np.str(dcoeff) + ' N = ' + np.str(positive_N)]))
##                
##                pop, prt = scipy.optimize.curve_fit(func_exp_pure,df_test['N'].astype(np.float),df_test['p_value'].astype(np.float), method = 'trf')
##                perr = np.sqrt(np.diag(prt))
##                print(pop, perr)
##                splot.plot(L[:,0], func_exp_pure(L[:,0], pop[0], pop[1]),  color = ctheme[c], linestyle='--')
##                labels = np.concatenate((labels, ['exponential']))                
#                
#    splot.tick_params(labelsize = fs)
#    splot.legend(labels[1:], bbox_to_anchor=(1, 1),ncol = 1,fancybox=True, shadow=True, fontsize = fs) # loc='best', 
##    splot.set_title(combination[np.str(c)], fontsize = fs)
#    splot.set_xlabel('Sample size (N)', fontsize = fs)
#    splot.set_ylabel('p_value ', fontsize = fs) # + combination[np.str(c)], fontsize = fs)
#    splot.set_ylim([-0.17,0.17])    
#    if save_fig == 1:
#        plt.savefig(path + combination[np.str(c)] + file_name, format=file_name[-3:], dpi=350) 
#    plt.show()
################################################################################

# -----------------------------------------------------------------------------
# --------- ANALYSIS OF ROBUSTNESS - effect of different grid size ------------
# -----------------------------------------------------------------------------
from exponential_fit import decission_data_exponential
from lowess_fit import decission_data_lowess
from decision_index import get_decision_index
import researchpy as rp


def grid_robustness(par1, par2, combination, measure, n0=None, ninf=None,
                    Nmax=None, nsize=None, k=None, initial_portion=None,
                    method=None):
    if n0 is None:
        n0 = 2
    if ninf is None:
        ninf = 2500
    if Nmax is None:
        Nmax = 10000  # number of datapoints available

    if nsize is None:
        nsize = 250  # grid size
    if k is None:
        k = 20
    if initial_portion is None:
        initial_portion = 1 / 3.

    # Cross validation parameters
    sign_level = 0.05
    gamma = 0.0001

    for i in range(0, 100):
        df = compare_dist(par1, par2, n0=n0, ninf=ninf, nsize=nsize,
                          k=k, initial_portion=initial_portion)

        # Compute the decision analysis of the estimated p-values
        if method == 'lowess':
            data_aux = decission_data_lowess(df, combination, measure,
                                             sign_level=sign_level,
                                             gamma=gamma)

        elif method == 'exponential' or method is None:
            data_aux = decission_data_exponential(df, combination, measure,
                                                  sign_level=sign_level,
                                                  gamma=gamma)

        # Calculate the decision index

        if i == 0:
            aux_theta = get_decision_index(data_aux, measure, combination)
            Theta = aux_theta
        else:
            aux_theta = get_decision_index(data_aux, measure, combination)
            Theta = pd.concat([Theta, aux_theta])
        print('Iteration ' + np.str(i) + '/100 succeed. Resulting Theta: ' + np.str(aux_theta['normal Theta'][0]))
    #        if aux_theta['normal Theta'][0] == 0:
    #            print('a: ' + np.str(data_aux.normal_exp_params[0][0]) + ', c: ' + np.str(data_aux.normal_exp_params[0][1]))
    #            plt.figure(figsize=(15,3))
    #            sns.boxplot(x='N', y='p_value', data = df)
    #            plt.show()
    result = np.zeros((len(measure), len(combination)))
    for m in range(len(measure)):
        for c in range(len(combination)):
            aux = Theta[Theta['comparison'] == combination[np.str(c)]][measure[np.str(m)] + ' Theta']
            result[m, c] = np.mean(aux[0]) * 100

    ## Summary of data
    aux = rp.summary_cont(df['p_value'].groupby(df['N']))
    #    print(len(aux))
    return Theta, result

# Tested conditions
normal_comparison={'0': 'normal_0_1_normal_0_1',
            '1': 'normal_0_1_normal_0.01_1',
            '2': 'normal_0_1_normal_0.1_1',
            '3': 'normal_0_1_normal_0.25_1',
            '4': 'normal_0_1_normal_0.5_1',
            '5': 'normal_0_1_normal_0.75_1',
            '6': 'normal_0_1_normal_1_1',
            '7': 'normal_0_1_normal_1.5_1',
            '8': 'normal_0_1_normal_2_1',
            '9': 'normal_0_1_normal_2.5_1',
            '10': 'normal_0_1_normal_3_1',
        }
measure={'0': 'normal'}
par1 = {
  "dist": "Normal",
  "mean": 0,
  "std": 1
}
par2 = {
  "dist": "Normal",
  "mean": 0,
  "std": 1
}
mean_values = [0,0.01,0.1,0.25,0.5,0.75,1,1.5,2,2.5,3]
n0 = 20
ninf = 2500
nsize = [10, 20, 50, 100, 150, 200]
portion = [1,1/2., 1/3., 1/5., 1/10.] # , 1/50., 1/100.]
k1 = 1
k2 = 20
data4excel = pd.DataFrame()

np.random.seed(seed=23456)

#for i in range(2,3):# , len(mean_values)):
#    par2['mean'] = mean_values[i]
#    aux2 = pd.DataFrame()
#    for s in range(len(nsize)):
#        for p in range(len(portion)):
#            print(nsize[s],portion[p])            
#            k = np.max((k2*portion[p],1))
#            initial_portion = k1*portion[p]  
##            df = compare_dist(par1, par2, n0=n0, ninf=ninf, nsize=nsize[s],
##                                            k=k,
##                                            initial_portion=initial_portion)
#            comparison = {'0': normal_comparison[np.str(i)]}
#            print(comparison['0'])
#            Theta, result = grid_robustness(par1, par2, comparison, measure, 
#                                            n0=n0, ninf=ninf, nsize=nsize[s],
#                                            k=k,
#                                            initial_portion=initial_portion,
#                                            method='exponential')
#            print(result)
#            aux = {'reduction':portion[p], 'grid_size': nsize[s],
#                   comparison['0']: result[0,:]}
#            aux1 = pd.DataFrame(data=aux)
#            aux2 = pd.concat([aux1,aux2])
#    if i == 2:
#        data4excel = aux2
#    else:
#        data4excel = pd.merge(data4excel,aux2,'right')
#data4excel.to_excel("grids_robustness_theoretical_2.xlsx")

## -----------------------------------------------------------------------------
## --------- ANALYSIS OF ROBUSTNESS - effect of different gamma values ---------
## -----------------------------------------------------------------------------

##gamma = [0.001]
##results = pd.DataFrame()
##for g in gamma:
##    aux = pd.DataFrame()
##    print('Gamma value '+ np.str(g))
##    print('Decission parameters for cell values')
##    decission_param_cell = decission_data_exponential(df_cell, combination, measure_cell, sign_level = 0.05, gamma = g)
##    Theta_cell = get_decision_index(decission_param_cell, measure_cell, combination)
##    print(Theta_cell)
##    decission_param_cell = decission_param_cell.drop(['test'], axis=1)
##    for m in measure_cell:
##        decission_param_cell = decission_param_cell.drop([measure_cell[m] +'_convergence_N'], axis=1)
##    decission_param_cell['gamma'] = g
##    aux=pd.merge(decission_param_cell,Theta_cell,'right')
##
##    print(decission_param_cell)
##
##    print('Decission parameters for cellular protrusions values')
##    decission_param_protrusion = decission_data_exponential(df_protrusions, combination, measure_protrusions, sign_level = 0.05, gamma = g)
##    Theta_prot = get_decision_index(decission_param_protrusion, measure_protrusions, combination)
##    print(Theta_prot)
##    decission_param_protrusion = decission_param_protrusion.drop(['test'], axis=1)
##    for m in measure_protrusions:
##        decission_param_protrusion = decission_param_protrusion.drop([measure_protrusions[m] +'_convergence_N'], axis=1)
##    decission_param_protrusion['gamma'] = g
##    aux_prot=pd.merge(decission_param_protrusion,Theta_prot,'right')
##    aux=pd.merge(aux,aux_prot,'right')
##
##
##    print('Decission parameters for cellular binary variable')
##    decission_param_binary = decission_data_exponential(df_binary, combination, measure_binary, sign_level = 0.05, gamma = g)
##    Theta_bin = get_decision_index(decission_param_binary, measure_binary, combination)
##    print(Theta_bin)
##    decission_param_binary = decission_param_binary.drop(['test'], axis=1)
##    for m in measure_binary:
##        decission_param_binary = decission_param_binary.drop([measure_binary[m] +'_convergence_N'], axis=1)
##    decission_param_binary['gamma'] = g
##    aux_bin=pd.merge(decission_param_binary,Theta_bin,'right')
##    aux=pd.merge(aux,aux_bin,'right')
##    results = pd.concat([results,aux])
##
##results.to_excel("theta_robustness.xlsx")
