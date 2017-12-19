#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 weshare.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: sas2csv.py
Author: zhushiliang(zhushiliang@baidu.com)
Date: 2017/10/24 11:41:39
"""
from sas7bdat import SAS7BDAT
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import multiprocessing
import logging
import os
import argparse


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
    
##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option], convert all sas7bdat file in a folder to csv')
    requiredgroup = parser.add_argument_group('required arguments')
    requiredgroup.add_argument('--path', dest='path', help='path of dir', default='', required=True)
    parser.add_argument('--worker',dest='worker',help='worker for multiprocess, 0 mean all', default=0, type=int)
    args = parser.parse_args()

    return args

def sas2csv(filename):
    logger.debug("dealing file:%s" % (str(filename)))
    with SAS7BDAT(filename) as f:
        df = f.to_data_frame()
        df.to_csv(filename+'.csv',index=False)
    logger.debug("file--%s done!" % (str(filename)))

def main_single():
    for f in os.listdir():
        if "sas7bdat" in f:
            sas2csv(f)
            break

def main():
    options = getOptions()
    
    if options.worker == 0:
        max_workers =  multiprocessing.cpu_count() - 1
    else:
        max_workers = options.worker

    with ProcessPoolExecutor(max_workers) as executor:
        jobs = {}

        for f in os.listdir(options.path):
            if "sas7bdat" in f:
                file = os.path.join(options.path, f)
                executor.submit(sas2csv,file)
                #jobs[executor.submit(self.mask_day,d)] = d

if __name__ == '__main__':
    main()

