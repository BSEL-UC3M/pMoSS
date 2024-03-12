# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:46:06 2019

@author: egomez
"""
import pandas as pd
import numpy as np
import os

def load_pvalue_data(name, path = None):
    if path is None:
        path = os.getcwd()
    if name == 'cell morphology':
        measure = {
                          '0': 'Cell body size microns',
                          '1': 'Cell body perimeter microns',
                          '2': 'Cell body flatness',
                          '3': 'Cell body roundness',
                          '4': 'Cell body axis ratio',
                          '5': 'protrusion_number'
                        }
        test = {  '0': 'MannWhitneyU',
                  '1': 't-test',
                }
#            cell_morphology = np.load('morphological_measures_p_value_new.npy')
        cell_morphology = np.load(os.path.join(path, 'morphological_measures_p_value_new_total.npy'))
        df = pd.DataFrame()
        df['p_value'] = cell_morphology[:,0]
        df['measure'] = cell_morphology[:,1]
        df['N'] = cell_morphology[:,2]
        df['comparison'] = cell_morphology[:,3] 
        df['test'] = cell_morphology[:,4]
        del cell_morphology
    elif name == 'protrusion_morphology':
        measure = {
                          '0': 'area_mu**2',
                          '1': 'perimeter_mu',
                          '2': 'length',
                          '3': 'diameter'
                        }
        test = {  '0': 'MannWhitneyU',
                  '1': 't-test',
                }
        protrusion_morphology = np.load(os.path.join(path, 'p_values_prot_morpho_new.npy'))
        df = pd.DataFrame()
        df['p_value'] = protrusion_morphology[:,0]
        df['measure'] = protrusion_morphology[:,1]
        df['N'] = protrusion_morphology[:,2]
        df['comparison'] = protrusion_morphology[:,3] 
        df['test'] = protrusion_morphology[:,4]
        del protrusion_morphology
    elif name == 'protrusion_binary':
        measure = {  '0': 'protrusion_binary'}
        test = {  '0': 'ChiSquared'}
        protrusions_binary = np.load(os.path.join(path, 'p_values_prot_number_binary_new.npy'))
        df = pd.DataFrame()
        df['p_value'] = protrusions_binary[:,0]   
        df['N'] = protrusions_binary[:,1] 
        df['comparison'] = protrusions_binary[:,2]   
        df['test'] = 'ChiSquared'
        df['measure'] = 'protrusion_binary'
        del protrusions_binary
    return df, measure, test


# def info_cells(data, cell_line):
#    cell_morpho ={'0': 'Cell body size microns',
#                        '1': 'Cell body perimeter microns',
#                        '2': 'Cell body flatness',
#                        '3': 'Cell body roundness',
#                        '4': 'Cell body axis ratio',
#                        '5': 'protrusion_number'}
#    df_cell = pd.DataFrame()
#    if cell_line == 'mammalian':# Mammalian cells (Praful)
#        df_cell[cell_morpho[str(0)]] = data[:,4].astype(float32)
#        df_cell[cell_morpho[str(1)]] = data[:,2].astype(float32)
#        df_cell[cell_morpho[str(2)]] = data[:,1].astype(float32)
#        df_cell[cell_morpho[str(3)]] = data[:,3].astype(float32)
#        df_cell[cell_morpho[str(4)]] = data[:,0].astype(float32)
#        df_cell[cell_morpho[str(5)]] = data[:,7]
#        df_cell['Condition'] = data[:,5]
#        df_cell['Video'] = data[:,6]
#    elif cell_line == 'glioblastoma':# Glioblastoma cells
#        for i in range(len(cell_morpho)):
#            df_cell[cell_morpho[str(i)]] = data[:,i].astype(float32)
#        df_cell['Condition'] = data[:,i+1]
#        df_cell['Video'] = data[:,i+2]
#
#    df_cell['protrusion_binary'] = df_cell.protrusion_number.astype(int)
#    df_cell['protrusion_binary'][df_cell.protrusion_binary > 0.0] = 1
#    cell_morpho ={'0': 'Cell body size microns',
#                 '1': 'Cell body perimeter microns',
#                 '2': 'Cell body flatness',
#                 '3': 'Cell body roundness',
#                 '4': 'Cell body axis ratio',
#                 '5': 'protrusion_binary'}
#                 # '6': 'protrusion_binary'}
#    return df_cell, cell_morpho
#
# def info_protrusions(data, cell_line):
#    prot_morpho = {'0': 'area_mu**2',
#                      '1': 'perimeter_mu',
#                      '2': 'length',
#                      '3': 'diameter'}
#    df_prot = pd.DataFrame()
#    if cell_line == 'mammalian': # Mammalian cells (Praful)
#        df_prot['Condition'] = data[:,0]
#        df_prot['Video'] = data[:,1]
#        for i in range(len(prot_morpho)):
#            df_prot[prot_morpho[str(i)]] = data[:,i+2].astype(float32)
#
#    elif cell_line == 'glioblastoma':# Glioblastoma cells (Alexandra)
#        for i in range(len(prot_morpho)):
#            df_prot[prot_morpho[str(i)]] = data[:,i].astype(float32)
#        df_prot['Condition'] = data[:,i+1]
#        df_prot['Video'] = data[:,i+2]
#
#    return df_prot, prot_morpho


def morphoparam(file_name, path=None):
    # When data is loaded from a text or excel file, please let the group variable
    # be in the first columns.
    print(path)
    if path is None:
        path = os.getcwd()

    #    if datatype == 'cell':
    #        data = np.load(path + 'morphological_measures_cell.npy')
    #        data,data_features = info_cells(data, cell_line)
    #
    #    elif datatype == 'protrusions':
    #        data = np.load(path + 'morphological_measures_prot.npy')
    #        data,data_features  = info_protrusions(data, cell_line)
    #    elif datatype == 'dataframe':

    print(file_name)
    try:
        if file_name[-4:] == 'xlsx':
            data = pd.read_excel(os.path.join(path, file_name), sheet_name='Sheet1',
                                 header=0)

        elif file_name[-3:] == 'csv':
            data = pd.read_csv(os.path.join(path, file_name), sep=';', header=0)

        variables = data.columns[1:]
        data_features = dict()

        for i in range(len(variables)):
            data_features[str(i)] = variables[i]

            # Obtain the labels of all the conditions to analyze
        group_labels = data.columns[0]
        group_labels = data.groupby([group_labels]).count()
        group_labels = group_labels.index.values
        keys = np.arange(len(group_labels)).astype(str)
        group_labels = dict(zip(keys, group_labels))

    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
        print(path + file_name + ' data does not exist.')

    return data, data_features, group_labels