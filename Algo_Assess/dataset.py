# -*- coding: utf-8 -*-

import config
import numpy as np
import pandas as pd
import gzip
import shutil



def skin_noskin_dataset(path_data_set):
    dataset = pd.read_csv(path_data_set, sep='\t')
    dataset = dataset.reindex(np.random.permutation(dataset.index))
    X = dataset.iloc[:, 0:3].values
    y = dataset.iloc[:,3].values
    
    # data cleaning
    y = [yi -1 for yi in y]
    
    return (X, y)


def red_wine_dataset(path_data_set):
    dataset = pd.read_csv(path_data_set, sep=';')
    dataset = dataset.reindex(np.random.permutation(dataset.index))
    X = dataset.iloc[:, 0:11].values
    y = dataset.iloc[:,11].values
    
    # data cleaning
    y = [yi -1 for yi in y]
    
    return (X, y)

def white_wine_dataset(path_data_set):
    dataset = pd.read_csv(path_data_set, sep=';')
    dataset = dataset.reindex(np.random.permutation(dataset.index))
    X = dataset.iloc[:, 0:11].values
    y = dataset.iloc[:,11].values
    
    # data cleaning
    y = [yi -1 for yi in y]
    
    return (X, y)