# -*-coding: utf-8-*-
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import *
import copy

##########################################################
# kexin needed
from sklearn_pandas import DataFrameMapper
from sklearn2pmml.decoration import CategoricalDomain, ContinuousDomain
from sklearn2pmml.preprocessing import ExpressionTransformer, PMMLLabelBinarizer, PMMLLabelEncoder
#from sklearn2pmml.feature_extraction.tabular import FeatureBinarizer
from sklearn.preprocessing import LabelBinarizer
##########################################################


class standard_feature_kexin(object):
    def __init__(self,data,target):
        self.data = data
        self.target = target
        
    def categ_continue_auto(self):
        self.continuousDomain = []
        self.categoricalDomain = []
        for item in self.data.columns:
            if (self.data[item].dtypes == object)| (self.data[item].dtypes == bool):
                self.categoricalDomain.append(item)
                self.data[item] = self.data[item].astype(str)
            #elif item!=target:                       # 去掉了target列
            else:
                self.continuousDomain.append(item)
                self.data[item] = self.data[item]
                
        # for testing
        #self.categoricalDomain = ['house_type', 'car_flg']
        #self.data['target_1'] = 1
        #self.continuousDomain = ['target_1']
                
    def only_continue(self):
        self.continuousDomain = []
        self.categoricalDomain = []
        for item in self.data.columns:
            self.continuousDomain.append(item)
            self.data[item] = self.data[item]
            
    def categ_trans(self):
#         self.dfm = DataFrameMapper([(c, [CategoricalDomain(invalid_value_treatment = 'as_missing',
#                                               missing_value_treatment = 'as_value',
#                                               missing_value_replacement = 'CreditX-NA'), FeatureBinarizer()])
#                        for c in self.categoricalDomain] + 
#                      [(self.continuousDomain, [ContinuousDomain(invalid_value_treatment = 'as_missing',
#                                                      missing_value_treatment = 'as_value',
#                                                      missing_value_replacement = np.nan)])],
#                      df_out = True)  # sklearn2pmml update and can not found FeatureBinarizer
        #self.data = self.data[['target','char403']]
        self.dfm = DataFrameMapper([(c, [CategoricalDomain(invalid_value_treatment = 'as_missing',
                                              missing_value_treatment = 'as_value',
                                              missing_value_replacement = 'CreditX-NA'),LabelBinarizer()])
                       for c in self.categoricalDomain] + 
                     [(c, ContinuousDomain(invalid_value_treatment = 'as_missing',
                                                     missing_value_treatment = 'as_value',
                                                     missing_value_replacement = -1))
                       for c in self.continuousDomain], df_out = True)
#         self.dfm = DataFrameMapper([(c, LabelBinarizer()) for c in self.categoricalDomain]+
#                                    [(c, None) for c in self.continuousDomain],
#                                    df_out = True)
        self.feature_transed = self.dfm.fit_transform(self.feature_imputed)

        
    def miss_inf_trans(self):
        self.imp = Imputer(missing_values='NaN', strategy='mean', axis=0,verbose=1)
        feature_imputed = self.imp.fit_transform(self.data[self.continuousDomain])
        feature_imputed_df = pd.DataFrame(feature_imputed, columns=self.continuousDomain)
        self.data.fillna('None')
        self.feature_imputed = pd.concat([self.data[self.categoricalDomain], feature_imputed_df],axis=1)

    def format_train_test(self,test_size=0.3,random_state=1992):
        self.feature_transed[self.target] = self.feature_transed[self.target].astype(int)
        self.Train, self.Test = train_test_split(self.feature_transed, test_size=test_size, random_state=random_state)

        self.sample_y = self.Train[self.target]
        self.sample_x = self.Train.drop(self.target,axis=1)
        self.test_y = self.Test[self.target]
        self.test_x = self.Test.drop(self.target,axis=1)        

    def apply_standardscale_classification(self,test_size=0.3,random_state=1992):
        self.format_train_test(test_size=test_size, random_state=random_state)
        self.scaler = StandardScaler().fit(self.sample_x)
        self.scaled_sample_x = self.scaler.transform(self.sample_x)
        self.scaled_test_x = self.scaler.transform(self.test_x)
        
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
