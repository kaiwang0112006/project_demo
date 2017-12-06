# -*-coding: utf-8-*-

import pandas as pd
from tools.features_engine import *
from tools.evaluate import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.pipeline import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FeatureUnion
from sklearn import datasets
import logging
logging.basicConfig(level=logging.DEBUG)
import time

def feature_engine_test():
    ###################################################################################
    # make data
    #data = pd.read_csv('/Users/wangkai/Documents/eclipse_workspace/pythondaily/mykaggle_prudential/prudential/train.csv')
    #data['Response'] = data['Response'].apply(lambda x:1 if x>4 else 0)
    df = pd.DataFrame(np.random.randn(5, 3), index=['a', 'c', 'e', 'f', 'h'],columns=['Product_Info_2','Ins_Age', 'three'])
    df['four'] = 'bar'
    # add commit 
    df['Response'] = df['Product_Info_2'] > 0
    data = df.reindex(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    data['four'] = data['four'].fillna('missing')
    data['Ins_Age']['c'] = -np.inf 
    data['Product_Info_2']['a'] = np.inf
    data['four']['b']='foo'
    print(data)
    print('\n')
    ###################################################################################
    # data preprocess version 1
    # 判断连续和类别型特征
    categoricalDomain, continuousDomain = categ_continue_auto_of_df(data,'Response')
    #print(categoricalDomain, continuousDomain)
    
    # 串行流水作业 version 1
    step1 = ('infinite', InfClass(continuous=continuousDomain,method='max_min')) # 正负无穷大处理为最大最小值
    step2 = ("imputer", ImputerClass(continuous=continuousDomain,strategy='mean'))  # 连续变量缺失值处理
    step3 = ('onehot', OneHotClass(catego=categoricalDomain, miss='missing'))

    pipeline = Pipeline(steps=[step1,step2,step3])
    newdata = pipeline.fit_transform(data)
    #print(newdata.head())
    
    ###################################################################################
    # data preprocess version 2
    # 判断连续和类别型特征
    #print(categoricalDomain, continuousDomain)
    
    # 串行流水作业 version 2
    step1 = ('infinite', InfClass(continuous=continuousDomain,method='max_min')) # 正负无穷大处理为最大最小值
    step2 = ("imputer", ImputerClass(continuous=continuousDomain,strategy='mean'))  # 连续变量缺失值处理
    step3 = ('label_encode', label_encoder_sk(cols=categoricalDomain))

    pipeline = Pipeline(steps=[step1,step2,step3])
    newdata = pipeline.fit_transform(data)
    #print(newdata.head())
    
    ###################################################################################
    # data preprocess version 3
    # 判断连续和类别型特征
    categoricalDomain, continuousDomain = categ_continue_auto_of_df(data,'Response')
    #print(categoricalDomain, continuousDomain)

    # 串行流水作业 version 3minmaxScalerClass
    step1 = ('infinite', InfClass(continuous=continuousDomain,method='max_min')) # 正负无穷大处理为最大最小值
    step2 = ("imputer", ImputerClass(continuous=continuousDomain,strategy='mean'))  # 连续变量缺失值处理
    step3 = ('onehot', OneHotClass(catego=categoricalDomain, miss='missing'))
    step4 = ('MinMaxScaler', minmaxScalerClass(cols=[],target="Response"))

    pipeline = Pipeline(steps=[step1,step2,step3,step4])
    newdata = pipeline.fit_transform(data)

    print(newdata.head())

    
def main():
    df = datasets.load_breast_cancer()
    ivobj = iv_pandas()
    datadf = pd.DataFrame(df.data,columns=[str(i) for i in range(30)])
    datadf['target'] = df.target
    #print(datadf.head())
    x=datadf['1']

    woe, iv = ivobj.cal_woe_iv(datadf,['1','2'],'target',nsplit=10,event=1)
    print(woe)
    print(iv)
    
    
if __name__ == '__main__':
    main()
