# -*-coding: utf-8-*-
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import *
import copy
from sklearn.base import *
from sklearn.pipeline import FeatureUnion
from sklearn.externals.joblib import Parallel, delayed
from scipy import sparse
import copy
import math
import itertools
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
    def __init__(self,cols,target='',t='minmax'):
        '''
        归一化类
        :param cols:
        :param target:
        :param t: 类型, type='minmax'是(x-min)/(xmax-xmin)归一化，type='std'是
                        (x-mean)/std方式
        '''
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

class f_combination():
    def __init__(self,nums=[1,2,3]):
        '''Exhaustive Feature Combination on one dimention

        :param nums:sorted list.
                    list of feature numbers. [1,2] means only generate new features based on
                    one or two old features

        '''
        self.nums = sorted(nums)


    def generate(self,df,keep=[],use=[]):
        '''

        :param df: dataframe. original dataframe
        :param keep: list. features that do not involve in generating new features
        :param addm: dict. added method expect default generation methods
                     key is the num in self.nums, value is a list with method name and
                     method function
                     example: {1:['method1':max]}
        :param use: feature to user for combination
        :return: new dataframe with generated features
        '''

        self.keep = keep
        if len(use)==0:
            self.features = [f for f in df if f not in keep]
        else:
            self.features = copy.deepcopy(use)
        #self.addm = addm
        newdf = copy.deepcopy(df)

        nummap = {1:self.__onecomb,2:self.__twocomb,3:self.__threecomb}

        for i in self.nums:
            newdf = nummap[i](newdf)
        return newdf

    def __onecomb(self,df):

        mdict = {
            'log':self.trylog,
            'power2':self.power2,
            'power3':self.power3
        }

        for f in self.features:
            for m in mdict.keys():
                newfeature = f + "_%s" % m
                df[newfeature] = df[f].apply(lambda x:mdict[m](x))
                break
        return df

    def __twocomb(self,df):
        for combs in itertools.combinations(self.features, 2):

            newfeature = "%s*%s" % (combs[0],combs[1])

            df[newfeature] = df[combs[0]] * df[combs[1]]
            #df[newfeature] = df.apply(lambda x: x[combs[0]]*x[combs[1]],axis=1)
        return df

    def __threecomb(self,df):
        for combs in itertools.permutations(self.features, 3):
            newfeature = "(%s+%s)*%s" % (combs[0],combs[1],combs[2])

            df[newfeature] = (df[combs[0]]+df[combs[1]])*df[combs[2]]
            #df[newfeature] = df.apply(lambda x: (x[combs[0]] + x[combs[1]] ) * x[combs[2]], axis=1)
        return df

    def power2(self,x):
        return x*x

    def power3(self,x):
        return x*x*x

    def twoby(self,x,y):
        return x*y

    def trylog(self,x):
        try:
            return math.log(x)
        except:
            return x