# -*-coding: utf-8-*-

import multiprocessing
import pandas as pd
import numpy as np
import logging
import traceback
try:
    from config import ENABLE_APPLY_BY_MULTIPROCESSING_MODE
except ImportError:
    ENABLE_APPLY_BY_MULTIPROCESSING_MODE = False


# Refer: `https://gist.github.com/yong27/7869662`

def apply_df(args):
    df, func, key, target,  kwargs = args
    
    df[target] = df[key].apply(func, **kwargs)
    
    return df 

def apply_row_series(args):
    df, func, kwargs = args
    datadf = df.apply(func, **kwargs, axis=1) # func need to return a pd.Series
    return datadf

def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers', multiprocessing.cpu_count() - 1)
    key = kwargs.pop('key')
    target = kwargs.pop('target')
    #with multiprocessing.Pool(processes=workers) as pool:
        
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(apply_df, [(d, func, key,target, kwargs)
                                  for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))

def apply_row_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers', multiprocessing.cpu_count() - 1)
    #with multiprocessing.Pool(processes=workers) as pool:
        
    pool = multiprocessing.Pool(processes=workers)
    result = pool.map(apply_row_series, [(d, func, kwargs)
                                  for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))
