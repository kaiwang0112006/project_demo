# -*-coding: utf-8-*-

import pandas as pd
from tools.features_engine import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.pipeline import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion
import logging
logging.basicConfig(level=logging.DEBUG)
import time

def main():
    ###################################################################################
    # make data
    #data = pd.read_csv('/Users/wangkai/Documents/eclipse_workspace/pythondaily/mykaggle_prudential/prudential/train.csv')
    #data['Response'] = data['Response'].apply(lambda x:1 if x>4 else 0)
    df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'h'],columns=['Product_Info_2','Ins_Age', 'three'])
    df['four'] = 'bar'
    
    df['Response'] = df['Product_Info_2'] > 0
    data = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    data['four'] = data['four'].fillna('missing')
    data['Ins_Age']['c'] = -np.inf 
    data['Product_Info_2']['a'] = np.inf
    data['four']['b']='foo'
    print(data)
    
    ###################################################################################
    # data preprocess
    # 判断连续和类别型特征
    categoricalDomain, continuousDomain = categ_continue_auto_of_df(data,'Response')
    print(categoricalDomain, continuousDomain)
    
    # 串行流水作业
    step1 = ('infinite', InfClass(continuous=continuousDomain,method='max_min')) # 正负无穷大处理为最大最小值
    step2 = ("imputer", ImputerClass(continuous=continuousDomain,strategy='mean'))  # 连续变量缺失值处理
    step3 = ('onehot', OneHotClass(catego=categoricalDomain, miss='missing'))

    pipeline = Pipeline(steps=[step1,step2,step3])
    newdata = pipeline.fit_transform(data)
    print(newdata.head())
    st = time.time()
    #pipeline = Pipeline(steps=[step3])
    pipeline.fit_transform(data)
    print('time ',time.time()-st)
    
  
    
if __name__ == '__main__':
    main()