# -*-coding: utf-8-*-
import pandas as pd
import numpy as np
from sklearn.metrics import *
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from collections import *
import os
from .information_value import *
import copy
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
        '''
        binning by split feature into equal n part (nsplit)
        '''
        self.woe = {}
        self.iv = {}
        df = df.reset_index(drop=True)
        for f in cols:
            woe, iv = self.calobj.woe_single_x(self.calobj.discrete(df[f],nsplit=nsplit),df[target],event=event)
            self.iv[f] = iv
            self.woe[f] = woe
        return self.woe, self.iv


def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    '''Calculate the PSI (population stability index) across all variables
    Args:
       expected: numpy matrix of original values
       actual: numpy matrix of new values, same size as expected
       buckettype: type of strategy for creating buckets, bins splits into even splits, quantiles splits into quantile buckets
       buckets: number of quantiles to use in bucketing variables
       axis: axis by which variables are defined, 0 for vertical, 1 for horizontal
    Returns:
       psi_values: ndarray of psi values for each variable
    Author:
       Matthew Burke
       github.com/mwburke
       worksofchart.com
    '''

    def psi(expected_array, actual_array, buckets):
        '''Calculate the PSI for a single variable
        Args:
           expected_array: numpy array of original values
           actual_array: numpy array of new values, same size as expected
           buckets: number of percentile ranges to bucket the values into
        Returns:
           psi_value: calculated PSI value
        '''

        def scale_range (input, min, max):
            input += -(np.min(input))
            input /= np.max(input) / (max - min)
            input += min
            return input


        breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack([np.percentile(expected_array, b) for b in breakpoints])

        expected_percents = np.histogram(expected_array, breakpoints)[0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            '''Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            '''
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return(value)

        psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))

        return(psi_value)

    if len(expected.shape) == 1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_values = np.empty(expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi(expected[:,i], actual[:,i], buckets)
        elif axis == 1:
            psi_values[i] = psi(expected[i,:], actual[i,:], buckets)

    return(psi_values)

class varible_exam:
    def __init__(self, value, target, missing=None):
        '''
        target: sample label
        value: varible value
        '''
        self.value = np.array(value)
        self.target = np.array(target)
        self.bindata = OrderedDict()
        self.npvalue = []
        self.missing = missing

    def binning(self, bins=10):
        '''
        bins: count of bin
        '''
        self.bindata = OrderedDict()
        value = self.value[self.value != self.missing]
        if self.missing != None:
            self.bindata[str(self.missing)] = {'range':str(self.missing),'total': len(self.value[self.value == self.missing])}
        hist, binr = np.histogram(value, bins=bins)
        for i in range(len(binr) - 1):
            start = binr[i]
            end = binr[i + 1] + 1 if i + 1 == len(binr) else binr[i + 1]
            self.bindata[(start, end)] = {'range':(start, end),'total': hist[i]}
        self.bindata['total'] = {'total': len(self.value),'range':'total'}
        return self

    def binning_with_range(self, bin_range):
        npvalue = self.value[self.value != self.missing]
        self.bindata = OrderedDict()
        if self.missing != None and len(self.value[self.value == self.missing])!=0:
            self.bindata[str(self.missing)] = {'range':str(self.missing),'total': len(self.value[self.value == self.missing])}
        for bins in bin_range:
            start = bins[0]
            end = bins[1]
            self.bindata[(start, end)] = {'range':(start, end),'total':len(npvalue[(npvalue >= start) & (npvalue < end)])}
        self.bindata['total'] = {'total': len(self.value), 'range':'total'}
        return self

    def good_bad_rate(self, good=1, bad=0):
        '''
        good: label for sample as 'good'
        bad: label for sample as 'bad'
        '''
        if len(self.bindata) == 0:
            self = self.binning()
        self.npvalue = self.value[self.value != self.missing]
        self.nptarget = np.array(self.target[self.value != self.missing])

        self.allgood = len(self.nptarget[(self.nptarget == good)])
        self.allbad = len(self.nptarget[(self.nptarget == bad)])

        for bin in self.bindata:
            if type(bin) == type((1, 2)):
                tg = self.nptarget[(self.npvalue >= bin[0]) & (self.npvalue < bin[1])]
            elif bin == 'total':
                tg = copy.deepcopy(np.array(self.target))
            elif type(bin) == type(''):
                tg = copy.deepcopy(self.target[self.value == self.missing])

            self.bindata[bin]['good'] = len(tg[tg == good])
            self.bindata[bin]['bad'] = len(tg[tg == bad])
            #self.bindata[bin]['total'] = len(tg)
            self.bindata[bin]['total%'] = len(tg) / len(self.value)
            try:
                self.bindata[bin]['odds'] = self.bindata[bin]['good'] / self.bindata[bin]['bad']
            except:
                self.bindata[bin]['odds'] = -1
            try:
                self.bindata[bin]['good%'] = self.bindata[bin]['good'] / (
                            self.bindata[bin]['good'] + self.bindata[bin]['bad'])
            except:
                self.bindata[bin]['good%'] = 0
            try:
                self.bindata[bin]['bad%'] = self.bindata[bin]['bad'] / (
                            self.bindata[bin]['good'] + self.bindata[bin]['bad'])
            except:
                self.bindata[bin]['bad%'] = 0

            try:
                self.bindata[bin]['agg_bad%'] = self.bindata[bin]['bad'] / self.allbad
            except:
                self.bindata[bin]['agg_bad%'] = 0

            try:
                self.bindata[bin]['agg_pass%'] = self.bindata[bin]['total'] / (len(self.value))
            except:
                self.bindata[bin]['agg_pass%'] = 0
        return self

    def woe_iv(self, good=1, bad=0):
        '''
        good: label for sample as 'good'
        bad: label for sample as 'bad'
        '''
        if len(self.npvalue) == 0:
            self = self.good_bad_rate(good, bad)
        iv = 0
        for bin in self.bindata:
            if type(bin) == type((1, 2)):
                tg = self.nptarget[(self.npvalue >= bin[0]) & (self.npvalue < bin[1])]
            elif bin == 'total':
                tg = copy.deepcopy(self.nptarget)
            elif type(bin) == type(''):
                tg = copy.deepcopy(self.target[self.value == self.missing])
            try:
                self.bindata[bin]['woe'] = math.log(
                    (self.bindata[bin]['bad'] / self.bindata[bin]['good']) / (self.allbad / self.allgood))
            except:
                self.bindata[bin]['woe'] = 0
            if bin == 'total':
                self.bindata[bin]['iv'] = iv
            else:
                try:
                    self.bindata[bin]['iv'] = (self.bindata[bin]['bad'] / self.allbad -
                                               self.bindata[bin]['good'] / self.allgood) * self.bindata[bin]['woe']
                    iv += self.bindata[bin]['iv']
                except:
                    self.bindata[bin]['iv'] = 0
        return self

if __name__=="__main__":
    #a = iv_pandas()
    #ksobj = ks_statistic(ytrue=[1, 0, 1, 0, 1, 0, 0], yprob=[0.9, 0.8, 0.7, 0.7, 0.6, 0.5, 0.4])

    #psiobj = psi()
    #p = psiobj.fit_mdl([1, 0, 1, 0, 1, 0, 0],box=False).cal([1, 0, 1, 0, 1, 1, 1])
    #print(p)
    import random
    vobj = varible_exam(list(range(100)),[random.choice([1, 0]) for i in range(100)])
    print(vobj.binning(10).good_bad_rate().woe_iv().bindata)
