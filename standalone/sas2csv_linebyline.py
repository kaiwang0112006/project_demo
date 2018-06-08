#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 weshare.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: sas2csv.py
Author: wangkai
Date: 2017/10/24 11:41:39
Note: do task as sas file to csv with less memory cost
"""

import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import multiprocessing
import logging
import os
import argparse
from project_demo.tools.sasParse import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
    
##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option], convert all sas7bdat file in a folder to csv')
    requiredgroup = parser.add_argument_group('required arguments')
    requiredgroup.add_argument('--input', dest='input', help='sas file', default='', required=True)
    args = parser.parse_args()

    return args

def sas2csv(filename):
    logger.debug("dealing file:%s" % (str(filename)))
    with SAS7BDAT_free(filename) as f:
        outname = filename+'.csv'
        f.linebyline2file(outname)
    logger.debug("file--%s done!" % (str(filename)))

def main_single():
    for f in os.listdir():
        if "sas7bdat" in f:
            sas2csv(f)
            break

def main():
    options = getOptions()
    sas2csv(options.input)
    

if __name__ == '__main__':
    main()

