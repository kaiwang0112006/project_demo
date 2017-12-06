# -*-coding: utf-8-*-
import pandas as pd
import numpy as np
from sklearn.metrics import *
from scipy.stats import ks_2samp
from .information_value import *

class ks_statistic(object):
    '''
    self understanding and ways to calculate ks
    '''
    def __init__(self,yprob,ytrue):
        '''
        yprob,ytrue are list
        '''
        self.yprob = yprob
        self.ytrue = ytrue
        
    def cal_ks_bak(self):
        '''
        get ks value
        '''
        
        preddf = pd.DataFrame({"ytrue":self.ytrue,"yprob":self.yprob})
        preddf.sort_values(by='yprob', axis=0, ascending=False, inplace=True)
        #print(preddf)
        ks_vs = []
        TPR = []
        FPR = []
        
        judgevalue = [min(self.yprob)]
        xp = max(self.yprob)
        for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            judgevalue.append(xp*i)
        for j in judgevalue:
            subdfpositive = preddf[preddf['ytrue']==1]
            subdfpositive['prd'] = subdfpositive.apply(lambda row:1 if (row['yprob']>=j) else 0,axis=1)
            tpr = len(subdfpositive[(subdfpositive['prd']==1) & (subdfpositive['ytrue']==1)])/len(subdfpositive)
            #print(subdfpositive)
            subnegative = preddf[preddf['ytrue']==0]
            subnegative['prd'] = subnegative.apply(lambda row:1 if (row['yprob']>=j) else 0,axis=1)
            fpr = len(subnegative[(subnegative['prd']==1) & (subnegative['ytrue']==0)])/len(subnegative)
            
            TPR.append(tpr)
            FPR.append(fpr)
            
        kss = [TPR[i]-FPR[i] for i in range(len(FPR))]
        return max(kss)
            
    def cal_ks(self,pos_label=1):
        self.fpr, self.tpr, thresholds = roc_curve(y_score=self.yprob, y_true=self.ytrue, pos_label=pos_label)

        kss = [abs(self.tpr[i]-self.fpr[i]) for i in range(len(self.fpr))]

        self.ks = max(kss)
        
def cal_ks_scipy(y_pred, y_true):
    '''
    cal ks using scipy
    '''
    return ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic

class iv_pandas(object):
    '''
    对IV计算的工具封装成适用于pandas输入, 返回结果一个是woe，格式为：{feature:{分箱id:woe}}, 另一个是iv，格式为 {feature:iv值}
    '''
    def __init__(self):
        self.calobj = WOE()

    def cal_woe_iv(self,df,cols,target,nsplit=10,event=1):
        self.woe = {}
        self.iv = {}
        
        for f in cols:
            woe, iv = self.calobj.woe_single_x(self.calobj.discrete(df[f],nsplit=nsplit),df[target],event=event)
            self.iv[f] = iv
            self.woe[f] = woe
        return self.woe, self.iv
            