# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 19:39:28 2018

@author: egomez
"""
import numpy as np
import pandas as pd
import os

#def info_cells(data, cell_line):
#    cell_morpho ={'0': 'Cell body size microns',
#                        '1': 'Cell body perimeter microns',
#                        '2': 'Cell body flatness',
#                        '3': 'Cell body roundness',
#                        '4': 'Cell body axis ratio',
#                        '5': 'protrusion_number'}
#    df_cell = pd.DataFrame()
#    if cell_line == 'mammalian':# Mammalian cells (Praful)
#        df_cell[cell_morpho[np.str(0)]] = data[:,4].astype(np.float32)
#        df_cell[cell_morpho[np.str(1)]] = data[:,2].astype(np.float32)
#        df_cell[cell_morpho[np.str(2)]] = data[:,1].astype(np.float32)
#        df_cell[cell_morpho[np.str(3)]] = data[:,3].astype(np.float32)
#        df_cell[cell_morpho[np.str(4)]] = data[:,0].astype(np.float32)
#        df_cell[cell_morpho[np.str(5)]] = data[:,7]
#        df_cell['Condition'] = data[:,5]   
#        df_cell['Video'] = data[:,6] 
#    elif cell_line == 'glioblastoma':# Glioblastoma cells
#        for i in range(len(cell_morpho)):
#            df_cell[cell_morpho[np.str(i)]] = data[:,i].astype(np.float32)
#        df_cell['Condition'] = data[:,i+1]   
#        df_cell['Video'] = data[:,i+2]   
#
#    df_cell['protrusion_binary'] = df_cell.protrusion_number.astype(np.int)
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
#def info_protrusions(data, cell_line):
#    prot_morpho = {'0': 'area_mu**2',
#                      '1': 'perimeter_mu',
#                      '2': 'length',
#                      '3': 'diameter'}
#    df_prot = pd.DataFrame()
#    if cell_line == 'mammalian': # Mammalian cells (Praful)
#        df_prot['Condition'] = data[:,0]   
#        df_prot['Video'] = data[:,1] 
#        for i in range(len(prot_morpho)):
#            df_prot[prot_morpho[np.str(i)]] = data[:,i+2].astype(np.float32)
#        
#    elif cell_line == 'glioblastoma':# Glioblastoma cells (Alexandra)    
#        for i in range(len(prot_morpho)):
#            df_prot[prot_morpho[np.str(i)]] = data[:,i].astype(np.float32)
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
        if file_name[-4:]=='xlsx':
            data = pd.read_excel(path+file_name,sheet_name = 'Sheet1', 
                                 header=0)
            
        elif file_name[-3:]=='csv':
            data = pd.read_csv(path+file_name,sep=';',header=0)
            
        variables = data.columns[1:]
        data_features = dict()
        
        for i in range(len(variables)):
            data_features[np.str(i)] = variables[i]    
            
        # Obtain the labels of all the conditions to analyze
        group_labels = data.columns[0]
        group_labels = data.groupby([group_labels]).count()
        group_labels = group_labels.index.values
        keys = np.arange(len(group_labels)).astype(np.str)        
        group_labels = dict(zip(keys, group_labels))
        
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
        print(path+file_name+' data does not exist.')       
    
    return data, data_features, group_labels

