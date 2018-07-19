# -*-coding: utf-8-*-
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import *
import copy
from sklearn.base import *
from sklearn.pipeline import FeatureUnion, _fit_one_transformer, _fit_transform_one, _transform_one 
from sklearn.externals.joblib import Parallel, delayed
from scipy import sparse

import logging
logger = logging.getLogger(__name__)

        
class label_encoder(object):
    def fit_pd(self,df,cols=[]):
        '''
        fit all columns in the df or specific list. 
        generate a dict:
        {feature1:{label1:1,label2:2}, feature2:{label1:1,label2:2}...}
        '''
        if len(cols) == 0:
            cols = df.columns
        self.class_index = {}
        for f in cols:
            uf = df[f].unique()
            self.class_index[f] = {}
            index = 1
            for item in uf:
                self.class_index[f][item] = index
                index += 1
    
    def fit_transform_pd(self,df,cols=[]):
        '''
        fit all columns in the df or specific list and return an update dataframe.
        '''
        if len(cols) == 0:
            cols = df.columns
        newdf = copy.deepcopy(df)
        self.class_index = {}
        for f in cols:
            uf = df[f].unique()
            self.class_index[f] = {}
            index = 1
            for item in uf:
                self.class_index[f][item] = index
                index += 1
                
            newdf[f] = df[f].apply(lambda d: self.update_label(f,d))
        return newdf
    
    def transform_pd(self,df,cols=[]):
        '''
        transform all columns in the df or specific list from lable to index, return an update dataframe.
        '''
        newdf = copy.deepcopy(df)
        if len(cols) == 0:
            cols = df.columns
        for f in cols:
            if f in self.class_index:
                newdf[f] = df[f].apply(lambda d: self.update_label(f,d))
        return newdf
                
    def update_label(self,f,x):
        '''
        update the label to index, if not found in the dict, add and update the dict.
        '''
        try:
            return self.class_index[f][x]
        except:
            self.class_index[f][x] = max(self.class_index[f].values())+1
            return self.class_index[f][x]
        
class standard_feature_tree(object):
    def __init__(self,data,target):
        self.data = data
        self.target = target
        
    def categ_continue_auto(self):
        self.categoricalDomain = []
        self.continuousDomain = []
        
        self.le = label_encoder()
        for item in self.data.columns:
            if (self.data[item].dtypes == object)| (self.data[item].dtypes == bool):
                self.categoricalDomain.append(item) 
            else:
                self.continuousDomain.append(item)
                
    def only_continue(self):
        self.le = label_encoder()
        self.continuousDomain = []
        self.categoricalDomain = []
        for item in self.data.columns:
            self.continuousDomain.append(item)
            self.data[item] = self.data[item]

    
    def categ_label_trans(self):
        self.feature_imputed = self.le.fit_transform_pd(self.feature_imputed, self.categoricalDomain) 
        
    def miss_inf_trans(self):
        self.imp = Imputer(missing_values='NaN', strategy='mean', axis=0,verbose=1)
        feature_imputed = self.imp.fit_transform(self.data[self.continuousDomain])
        feature_imputed_df = pd.DataFrame(feature_imputed, columns=self.continuousDomain)
        self.data.fillna('None')
        self.feature_imputed = pd.concat([self.data[self.categoricalDomain], feature_imputed_df],axis=1)
        
    def miss_inf_not_tans(self):
        self.feature_imputed = copy.deepcopy(self.data) 

    def format_train_test(self,test_size=0.3,random_state=1992):
        self.feature_imputed[self.target] = self.feature_imputed[self.target].astype(int)
        self.Train, self.Test = train_test_split(self.feature_imputed, test_size=test_size, random_state=random_state)

        self.sample_y = self.Train[self.target]
        self.sample_x = self.Train.drop(self.target,axis=1)
        self.test_y = self.Test[self.target]
        self.test_x = self.Test.drop(self.target,axis=1)        

    def apply_standardscale_classification(self,test_size=0.3,random_state=1992):
        self.format_train_test(test_size=test_size, random_state=random_state)
        self.scaler = StandardScaler().fit(self.sample_x)
        self.scaled_sample_x = self.scaler.transform(self.sample_x)
        self.scaled_test_x = self.scaler.transform(self.test_x)


def categ_continue_auto_of_df(df, target):
    continuousDomain = []
    categoricalDomain = []
    for item in df.columns:
        if item==target:
            pass 
        elif (df[item].dtypes == object)| (df[item].dtypes == bool):
            categoricalDomain.append(item)
        else:
            continuousDomain.append(item)
    return categoricalDomain, continuousDomain

class OneHotClass(BaseEstimator, TransformerMixin):
    '''
    针对类别型变量做独热编码，并去掉NA对应的哑变量
    '''
    def __init__(self, catego, miss='NA'):
        self.miss = miss
        self.catego = catego
    
    def fit(self, X, y=None):
        data = pd.get_dummies(X, columns=self.catego)
        self.features = list(data.columns)
        return self
    
    def transform(self, X, y=None):
        data = pd.get_dummies(X, columns=self.catego)
        feature_keep = []
        for f in self.features:
            if f in data and f.split('_')[-1]!=self.miss:
                feature_keep.append(f)
        return data[feature_keep]
        


class ImputerClass(BaseEstimator, TransformerMixin):
    '''
    缺失值处理，missing_values指缺失值类型，strategy指替代策略，可以是平均值，中位数或者众数，也可以是
    具体要替换的值。
    
    >>> imp = ImputerClass(continuous, missing_values='NaN', strategy='mean')
    >>> cleandata = imp.fit(data)
    
    >>> imp = ImputerClass(continuous, missing_values='missing', strategy=0)
    >>> cleandata = imp.fit(data)    
    '''
    def __init__(self, continuous, missing_values='NaN', strategy='mean'):
        self.continuous = continuous
        self.strategy = strategy
        self.missing_values = missing_values
        self.imputer = Imputer(missing_values=missing_values, strategy=strategy)
        
    def fit(self, X, y=None):
        if self.strategy in ['mean', 'median', 'most_frequent']:
            self.imputer.fit(X=X[self.continuous], y=y)
        return self
    
    def transform(self, X, y=None):
        if self.strategy in ['mean', 'median', 'most_frequent']:
            X[self.continuous] = self.imputer.transform(X[self.continuous])
        else:
            X[self.continuous] = X[self.continuous].replace(self.missing_values,self.strategy)
        return X

class InfClass(BaseEstimator, TransformerMixin):
    '''
    正负无穷大处理，将无穷大替换为最大最小值（method='max_min'），或者替换为任意指，如 0（method＝0）
    '''
    def __init__(self, continuous, method="max_min"):
        self.continuous = continuous
        if method == 'max_min':
            self.maxmethod = np.max
            self.minmethod = np.min 
        else:
            self.maxmethod = lambda x:method
            self.minmethod = lambda x:method
        #logger.debug(self.minmethod)
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_copy = copy.deepcopy(X)
        X_copy[self.continuous] = X_copy[self.continuous].replace([np.inf, -np.inf], np.nan)
        
        for f in self.continuous:
            X[f] = X[f].replace(np.inf, self.maxmethod(X_copy[f]))
            X[f] = X[f].replace(-np.inf, self.minmethod(X_copy[f]))
        return X 

class label_encoder_sk(BaseEstimator, TransformerMixin):
    '''
    类sklearn的方法做label_encoder,如果fit给定的columns是空的，对所有列处理
    '''
    def __init__(self, cols):
        self.cols = cols
        self.class_index = {}
    
    def fit(self,X,y=None):
        '''
        fit all columns in the df or specific list. 
        generate a dict:
        {feature1:{label1:1,label2:2}, feature2:{label1:1,label2:2}...}
        '''
        if len(self.cols) == 0:
            self.cols = X.columns
        self.class_index = {}
        for f in self.cols:
            uf = X[f].unique()
            self.class_index[f] = {}
            index = 1
            for item in uf:
                self.class_index[f][item] = index
                index += 1
        return self
    
    def transform(self,X,y=None):
        '''
        transform all columns in the df or specific list from lable to index, return an update dataframe.
        '''
        newdf = copy.deepcopy(X)
        for f in self.cols:
            if f in self.class_index:
                newdf[f] = X[f].apply(lambda d: self.update_label(f,d))
        return newdf
                
    def update_label(self,f,x):
        '''
        update the label to index, if not found in the dict, add and update the dict.
        '''
        try:
            return self.class_index[f][x]
        except:
            self.class_index[f][x] = max(self.class_index[f].values())+1
            return self.class_index[f][x]
        
class minmaxScalerClass(BaseEstimator, TransformerMixin):
    def __init__(self,cols,target):
        self.cols = cols 
        self.target = target
        self.scaler = MinMaxScaler()
        
    def fit(self,X,y=None):
        if len(self.cols) == 0:
            self.cols = [f for f in X.columns if f!=self.target]
        self.scaler.fit(X[self.cols])
        return self 
    
    def transform(self,X,y=None):
        X[self.cols] = self.scaler.transform(X[self.cols])
        return X
        
