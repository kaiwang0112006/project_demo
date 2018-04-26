# -*-coding: utf-8-*-
import pandas as pd
import numpy as np
from sklearn.metrics import *
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import os 
#os.environ['QT_QPA_PLATFORM']='offscreen'

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
        
    def cal_ks_with_plot(self,file="ks.png",split=10,plot=False):
        if not plot:
            os.environ['QT_QPA_PLATFORM']='offscreen'
        df_score = pd.DataFrame(self.yprob)
        df_good = pd.DataFrame(self.ytrue) 
        df_score.columns = ['score']
        df_good.columns = ['good']
        df = pd.concat([df_score,df_good],axis=1)
        
        df['bad'] = 1 - df.good
        bin = np.arange(0, 1.001, 0.05)
        df['bucket'] = pd.qcut(df.score, split, duplicates='drop')
        #df['bucket'] = pd.cut(df.score, bin)  # 根据bin来划分区间

        grouped = df.groupby('bucket', as_index=False) # 统计在每个区间的样本量
        aggdf = pd.DataFrame()

        aggdf['min_scr'] = grouped.min().score # 取得每个区间的最小值
        aggdf['max_scr'] = grouped.max().score
        aggdf['bads'] = grouped.sum().bad # 计算每个区间bad的总数量
        aggdf['goods'] = grouped.sum().good
        
        aggdf = (aggdf.sort_values(['min_scr'])).reset_index(drop=True) # 根据区间最小值排序
        aggdf['bad_cum_rate'] = np.round((aggdf.bads / df.bad.sum()).cumsum(), 4) # 计算bad样本累计概率
        aggdf['good_cum_rate'] = np.round((aggdf.goods / df.good.sum()).cumsum(), 4) 
        aggdf['ks'] = abs(np.round(((aggdf.bads / df.bad.sum()).cumsum() - (aggdf.goods / df.good.sum()).cumsum()), 4)) # 计算bad和good累计概率之差的绝对值

        self.ks = aggdf.ks.max()  # 求出ks

        plt.figure()  # 创建绘图对象
        plt.plot(aggdf.min_scr, aggdf.bad_cum_rate, "g-", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
        plt.plot(aggdf.min_scr, aggdf.good_cum_rate, "b-", linewidth=1)
        
        x_abline = aggdf['min_scr'][aggdf['ks'] == aggdf['ks'].max()] # ks最大的min_scr
        y_abline1 = aggdf['bad_cum_rate'][aggdf['ks'] == aggdf['ks'].max()] # ks最大时bad_cum_rate
        y_abline2 = aggdf['good_cum_rate'][aggdf['ks'] == aggdf['ks'].max()]
        plt.fill_between(x_abline, y_abline1, y_abline2, color = "red",linewidth=2)    
        
        sub = "%s%s"%('ks = ',self.ks)
        plt.legend(title=sub,loc='lower right')
        plt.xlabel("Minimum score")  # X轴标签
        plt.ylabel("Cumulative percentage(%)")  # Y轴标签
        plt.title('KS chart')  # 图标题
        if plot:
            plt.show()  # 显示图
        else:
            plt.savefig(file)
    
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
        df = df.reset_index(drop=True)
        for f in cols:
            woe, iv = self.calobj.woe_single_x(self.calobj.discrete(df[f],nsplit=nsplit),df[target],event=event)
            self.iv[f] = iv
            self.woe[f] = woe
        return self.woe, self.iv
    
if __name__=="__main__":
    ksobj = ks_statistic(ytrue=[1, 0, 1, 0, 1, 0, 0], yprob=[0.9, 0.8, 0.7, 0.7, 0.6, 0.5, 0.4])
    ksobj.cal_ks_with_plot(file="D:/test.png",plot=True)
            