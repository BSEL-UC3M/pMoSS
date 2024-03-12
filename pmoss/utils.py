# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 19:39:28 2018

@author: egomez
"""
import numpy as np
import glob

class group(object):
    def __init__(self, number, name):
        self.number = number
        self.group = name

def group_names(dir_name):
    # dir_name:  "/home/esgomezm/Projects/3D-PROTUCEL/glioblastoma/image_data/"
    
    dir_name = dir_name + "*"
    folder = glob.glob(dir_name)
    folder.sort()
    condition = []
    for f in range(len(folder)):
        condition = condition + [group(str(f),folder[f][len(dir_name)-1:])]
    condition = dict([ (c.number, c.group) for c in condition ])
    
    return condition

class group_combine(object):
    def __init__(self, number, groupA, groupB):
        self.number = number
        self.group = groupA + '_' + groupB
        
def group_combination(dir_name):
    # dir_name:  "/home/esgomezm/Projects/3D-PROTUCEL/glioblastoma/image_data/"
    condition = group_names(dir_name)

    count = 0
    combination = []
    for c in range(len(condition)):
        
        if c+1 < len(condition):
            
            for k in range(c+1,len(condition)):
                combination = combination + [group_combine(str(count),
                                            condition[str(c)],
                                            condition[str(k)])]
    
                count = count + 1

    combination = dict([ (c.number, c.group) for c in combination ])
    
    return combination


def create_combination(group_dict):
    count = 0
    combination = []
    keys = [k for k in group_dict]
    for c in range(len(group_dict)):
        if c+1 < len(group_dict):
            for k in range(c+1,len(group_dict)):
                combination = combination + [group_combine(str(count),
                                            group_dict[keys[c]], 
                                            group_dict[keys[k]])]
                count = count + 1

    combination = dict([ (c.number, c.group) for c in combination ])
    
    return combination