# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 11:38:47 2022

@author: egomez
"""

# Load the packages needed to run the scripts in this notebook
import os
import numpy as np
import pandas as pd
from pmoss.analysis import compute_diagnosis
from pmoss import create_combination
from pmoss.display import scatterplot_decrease_parameters, plot_pcurve_by_measure, composed_plot, table_of_results
from pmoss.models.exponential_fit import decission_data_exponential
from pmoss.loaders import morphoparam
# Avoid warnings
import warnings
warnings.filterwarnings('ignore')

# path containing the data
path = "/Users/esti/Documents/PROYECTOS/3DPROTUCELL/pmoss/pMoSS/data/morphology_taxol"
# Name of the file containing the information. It can be a csv or excel file.
# Note that the column containing the labels of the group must have the name "Condition"
# and should be the first column of the file.
# You can read either a csv or excel files:
file_name ='cell_data.xlsx'

# number of "n-values" to evaluate (size of N-grid)
grid_size = 10
# minimum "n-value" to compute Monte Carlo cross-validation
n0 = 2
# maximum "n-value" to compute Monte Carlo cross-validation
Nmax = 2500

# This value prevents from having only one iteration for the highest "n-value":
# final iterations = k*(m/min(m,Nmax)) where m is the size of group with less observations.
k = 5

# This value prevents from having millions of iterations in n0 (the lowest"n-value"):
# initial iterations = np.log((m/n0)*initial_portion) where m is the size of group with less observations.
initial_portion=0.00001

alpha = 0.05 # alpha for a 100(1-alpha) statistical significance.
gamma = 5e-06 # gamma in the paper = gamma*alpha.
# Statistitical test to evaluate
test = 'MannWhitneyU'
# Method to estimate the p-value function
method = 'exponential'
pvalues, param, Theta = compute_diagnosis(file_name, path = path, gamma = gamma,
                                          alpha = alpha, grid_size = grid_size,
                                          n0 = n0, Nmax = Nmax,k = k,
                                          initial_portion=initial_portion,
                                          method = method, test = test)
# Save computed parameters
pvalues.to_csv(os.path.join(path, 'cell_morphology_pvalues.csv'), index = False)

del pvalues

# Load the data
file_name = r'cell_data.xlsx'
df = pd.read_csv(os.path.join(path, 'cell_morphology_pvalues.csv'), sep=',')

# Obtain the data, variables and name of the groups for which you would like to get a plot
data, variables, group_labels = morphoparam(file_name, path=path)

# You can create all the combinations from a dictionary with the labels of each group, or declare which combinations you want:
# 1.- All combinations should be written exactly as in the csv of the p-values.
combination = create_combination(group_labels)

# Calculate the data related to exponential parameters:
param = decission_data_exponential(df, combination, variables, sign_level = 0.05, gamma = 5e-06)

# print the results:
table = table_of_results(param, variables, combination)
print("Table of results")
print("-------------------------------------------------")
print(table)
print(" ")

# Plot exponential parameters a and c from p(n) = aexp(-cn)
colors = ['#FF0000', '#F89800', '#0200DE']
scatterplot_decrease_parameters(df, combination,variables, path = path,fs = 10, width = 5, height = 5, plot_type="exp-param", colors = colors)

# Plot the estimator of the minimum sample size to observe statistically significant differences.
scatterplot_decrease_parameters(df, combination,variables, path = path,fs = 10, width = 5, height = 5, plot_type="sampled-nalpha", colors = colors)
# Plot the sample size n that satisfies alpha = aexp(-cn). This value is the theoretical minimum sample size needed to observe statistically significant differences.
scatterplot_decrease_parameters(df, combination,variables, path = path,fs = 10, width = 5, height = 5, plot_type="theory-nalpha", colors = colors)
# Plot the p-function for continuous measures
colors = ['#FF0000', '#F89800', '#0200DE']
continuous_variables = {i:variables[i] for i in variables if variables[i]!='protrusion_binary'}
plot_pcurve_by_measure(df, combination, continuous_variables, path = path, colors = colors)
# Plot the p-function for continuous measures
colors = ['#FF0000', '#F89800', '#0200DE']
continuous_variables = {i:variables[i] for i in variables if variables[i]!='protrusion_binary'}
composed_plot(data, df, group_labels, combination, continuous_variables, colors = colors,
              fs = 20, width = 32, height = 10, bins = 1500)
# Plot the p-function for discrete variables measures
discrete_variables = {'0': 'protrusion_binary'}
test={'0': 'ChiSquared'}
plot_pcurve_by_measure(df, combination, discrete_variables, path = path, test=test, colors = colors)
# Plot the p-function for discrete variables measures
discrete_variables = {'0': 'protrusion_binary'}
test={'0': 'ChiSquared'}
composed_plot(data, df, group_labels, combination, discrete_variables,test=test,
              colors = colors, fs = 20, width = 30, height = 10, bins = 5)