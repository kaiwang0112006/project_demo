import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


class OddsPlot(object):

    def __init__(
            self, feature_list, label_list, feat_name='feat_name', save_path=None, with_bin=True, bins=50, qcut=True):
        self.__df = pd.DataFrame({'feature': feature_list, 'label': label_list})
        self.__feat_name = feat_name
        self.__save_path = save_path
        self.__with_bin = with_bin
        self.__bins = bins
        self.__qcut = qcut
        self.__plot()

    def __plot(self):

        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['font.family']='sans-serif'
        matplotlib.rcParams['axes.unicode_minus'] = False
        plt.rcParams['savefig.dpi'] = 150
        plt.rcParams['figure.dpi'] = 150
        plot_df = self.__get_plot_df()

        fig = plt.figure(figsize=(20, 8))
        # 占比图
        ax1 = fig.add_subplot(121)
        x = [str(i) for i in plot_df.index]
        ax1.bar(x, plot_df['bin_n_rate'], width=0.5, bottom=0)
        x_tmp = np.arange(len(plot_df.index))
        ax1.bar(x_tmp+0.2, plot_df['bin_p_rate'], width=0.5, bottom=0)
        ax1.legend(['good', 'bad'])
        # odds图
        ax2 = fig.add_subplot(122)
        ax2.bar(x, plot_df['bin_odds'], width=0.5, bottom=0)
        ax2.legend(['good/bad'])
        # 显示设置
        fig.suptitle(self.__feat_name)
        if self.__with_bin:
            ax1.set_xticklabels(x, fontdict={'fontsize': 9, 'rotation': 30})
            ax2.set_xticklabels(x, fontdict={'fontsize': 9, 'rotation': 30})
        if self.__save_path:
            fig.savefig(self.__save_path)
        else:
            fig.show()

    def __get_plot_df(self):
        total_p = sum(self.__df['label'])
        total_n = self.__df.shape[0] - total_p
        if self.__with_bin:
            if self.__qcut:
                self.__df['bin'] = pd.qcut(self.__df['feature'], self.__bins, duplicates='drop')
            else:
                self.__df['bin'] = pd.cut(self.__df['feature'], self.__bins)
        else:
            self.__df['bin'] = self.__df['feature']
        group_by_obj = self.__df.groupby('bin')
        plot_df = pd.DataFrame(group_by_obj.count()['feature'])
        plot_df.rename(columns={'feature': 'bin_num'}, inplace=True)
        plot_df['bin_p'] = group_by_obj.sum()['label']
        plot_df['bin_n'] = plot_df['bin_num'] - plot_df['bin_p']
        plot_df['bin_p_rate'] = plot_df['bin_p'] * 1. / total_p
        plot_df['bin_n_rate'] = plot_df['bin_n'] * 1. / total_n
        plot_df['bin_odds'] = plot_df['bin_n'] * 1. / (plot_df['bin_p'] + 1)
        plot_df.fillna(0, inplace=True)
        return plot_df