# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
##########################################
## Options and defaults
##########################################
def getOptions():
    parser = argparse.ArgumentParser(description='python *.py [option], convert all sas7bdat file in a folder to csv')
    requiredgroup = parser.add_argument_group('required arguments')
    requiredgroup.add_argument('--file', dest='pdfile', help='csv file', default='', required=True)
    parser.add_argument('--split',dest='spnum',help='split file to how many parts', default=2, type=int)
    args = parser.parse_args()

    return args

def main():
    options = getOptions()
    
    basepath, filename = os.path.split(options.pdfile)
    if basepath == '':
        basepath = os.getcwd()
    
    df = pd.read_csv(options.pdfile)
    count = 1
    for pdf in np.array_split(df, workers):
        pname = filename.split('.')[0] + "_" + str(count) + '.' + filename.split('.')[1]
        pdf.to_csv(pname,index=False)
    

if __name__ == '__main__':
    main()