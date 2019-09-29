# -*- coding:utf-8 -*-
"""
@author: caoxi
@time: 2018/4/12 20:21
"""
import pandas as pd
import numpy as np
import math
import dill
import sys


class EqualWidthBinning(object):
    def __init__(self, list_p, list_label, bins, standard_score, standard_odds, pdo):
        self.__bins = bins
        self.__standard_score = standard_score
        self.__standard_odds = standard_odds
        self.__pdo = pdo
        self.__length = len(list_p)
        self.__step = 1.0 / self.__bins
        self.__list_level = range(self.__bins)
        self.__df_binning = pd.DataFrame(index=self.__list_level, columns=["SCR-L", "SCR-R"])
        self.__df_score = pd.DataFrame({"P-SCR": list_p, "LB": list_label})
        self.__df_mapping = None
        self.__res = None
        self.__binning_init()
        self.__fill_binning()

    @staticmethod
    def __create_fun(df):
        """
        通过两点确定一条直线，让整个映射连续
        :param df: 输入的Dataframe
        :return: 直线的斜截式
        """
        lx = df["L-X"]
        ly = df["L-Y"]
        rx = df["R-X"]
        ry = df["R-Y"]
        return lambda X: round((ry - ly) / (rx - lx) * X + (rx * ly - lx * ry) / (rx - lx))

    def __binning_init(self):
        """
        针对train集的分数进行第一次分箱
        """
        list_score_left = np.arange(0, 1, self.__step)
        list_score_right = np.arange(self.__step, 1 + self.__step, self.__step)
        self.__df_binning["SCR-L"] = list_score_left
        self.__df_binning["SCR-R"] = list_score_right
        self.__df_score["BIN"] = self.__df_score["P-SCR"].apply(lambda x: int(x / self.__step))

    def __fill_binning(self):
        """
        利用train分箱内的正负样本计算真实odds，得到odds的离散映射函数
        """
        bads = self.__df_score["LB"].groupby(self.__df_score["BIN"]).sum()
        hits = self.__df_score["LB"].groupby(self.__df_score["BIN"]).count()
        self.__df_binning["BAD-HITS"] = bads
        self.__df_binning["CUR-HITS"] = hits
        self.__df_binning["GOOD-HITS"] = self.__df_binning["CUR-HITS"] - self.__df_binning["BAD-HITS"]
        self.__df_binning["ODDS"] = self.__df_binning["GOOD-HITS"] / self.__df_binning["BAD-HITS"]
        self.__df_binning = self.__df_binning.fillna(0)
        self.__df_binning["ODDS"][np.isinf(self.__df_binning["ODDS"])] = 0
        try:
            max_odds = max(self.__df_binning[self.__df_binning["ODDS"] != 0]["ODDS"])
            min_odds = min(self.__df_binning[self.__df_binning["ODDS"] != 0]["ODDS"])
        except:
            print(self.__df_binning)
        is_zero_bad = False
        is_zero_good = False
        for i in range(int(self.__bins / 2)):
            # 从分布的中间往前处理bad-hits为0的情况， 前提要保证p值在0.5左右的时候bad-hits不为0
            if self.__df_binning["BAD-HITS"][self.__bins / 2 - 1 - i] == 0.0:
                is_zero_bad = True
            if is_zero_bad:
                max_odds *= 2
                self.__df_binning["ODDS"][self.__bins / 2 - 1 - i] = max_odds
            # 从分布的中间往后处理bad-hits为0的情况，前提同理
            if self.__df_binning["GOOD-HITS"][self.__bins / 2 + i] == 0.0:
                is_zero_good = True
            if is_zero_good:
                min_odds /= 2
                self.__df_binning["ODDS"][self.__bins / 2 + i] = min_odds / 2

    def get_mapping_df(self):
        """
        取train各分箱的平均数作为锚点，连接各锚点，生成连续的odds映射函数
        :return:  生成的mapping Dataframe
        """
        # print(self.__df_binning["ODDS"])
        self.__df_binning["Y-SCR"] = self.__df_binning["ODDS"] \
                .apply(lambda x: int(self.__standard_score + self.__pdo * (math.log(x+1) - math.log(self.__standard_odds))
                                     / math.log(2)))
        self.__df_binning["AV-SCR"] = (self.__df_binning["SCR-L"] + self.__df_binning["SCR-R"]) / 2
        self.__df_mapping = pd.DataFrame(index=range(self.__bins + 1), columns=["L-X", "R-X", "L-Y", "R-Y"])
        self.__df_mapping.fillna(0.0, inplace=True)
        self.__df_mapping["L-X"][1:self.__bins + 1] = self.__df_binning["AV-SCR"]
        self.__df_mapping["L-Y"][1:self.__bins + 1] = self.__df_binning["Y-SCR"]
        self.__df_mapping["R-X"][0:self.__bins] = self.__df_binning["AV-SCR"]
        self.__df_mapping["R-Y"][0:self.__bins] = self.__df_binning["Y-SCR"]
        self.__df_mapping["L-X"][0] = 0
        self.__df_mapping["R-X"][self.__bins] = 1
        self.__df_mapping["R-Y"][self.__bins] = min(self.__df_binning["Y-SCR"]) - 20
        self.__df_mapping["L-Y"][0] = max(self.__df_binning["Y-SCR"]) + 20
        self.__df_mapping["FUN"] = self.__df_mapping.apply(self.__create_fun, axis=1)
        # 如要保存mapping结果, 需要安装dill，pickle不支持lambda函数的序列化
        return self.__df_mapping

    def fill_score(self, stat_bins, p_list, list_stat_label, path=None):
        """
        画出输入数据的映射分数分布并选择是否保存
        :param stat_bins: 统计需要的分箱数
        :param list_stat_label: 输入数据的label向量
        :param p_list: 输入数据的p值向量
        :param path: 选择是否生成分布图或保存
        :return: 生成的分布映射Dataframe
        """
        stat_step = 1.0 / stat_bins
        df_score = pd.DataFrame({"P-SCR": p_list, "LB": list_stat_label})
        df_score["BIN"] = df_score["P-SCR"].apply(lambda x: int(x / stat_step))
        df_score["S-BIN"] = df_score["P-SCR"].apply(lambda x: int((x + self.__step / 2) / self.__step))
        df_res = pd.merge(left=df_score, right=self.__df_mapping, left_on="S-BIN", right_index=True)
        df_res["Y-SCR"] = df_res.apply(lambda x: x["FUN"](x["P-SCR"]), axis=1)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
            if path:
                sns.distplot(df_res["Y-SCR"], bins=self.__bins, kde=True, hist=True)
                plt.savefig(path, format='png', dpi=100)
                plt.close()
        except:
            pass
        return df_res

    @staticmethod
    def single_run(mapping_path, p):
        """
        :param mapping_path: 映射文件路径
        :param p: 原始的p值
        :return: 标准评分卡评分
        """
        dict_mapping = dill.load(open(mapping_path, 'rb'))
        length = len(dict_mapping["FUN"])
        step = 1.0/(length-1)
        bin_index = int((p + step / 2) / step)
        return dict_mapping["FUN"][bin_index](p)

    def get_binning_stat_df(self, p_list, list_stat_label, picture_path, stat_bins):
        """
        :param list_stat_label:
        :param p_list: 原始的p值向量
        :param picture_path: 保存的分布图路径
        :param stat_bins: 输入的统计分箱数
        :return: 等宽分箱后的统计表
        """
        df_res = self.fill_score(stat_bins, p_list, list_stat_label, picture_path)
        # print(df_res)
        df_stat = pd.DataFrame(
            data=0.0, index=range(stat_bins),
            columns=["SCR-AVG", "ODDS", "SCR-L", "SCR-R", "HITS", "BAD-HITS", "OD-RA", "CUM-PA", "CUM-OD", "RE-OD"])
        step = 1.0 / stat_bins
        p_left_list = np.arange(0, 1, step)
        score_right_list = np.arange(step, 1 + step, step)
        df_stat["SCR-L"] = p_left_list
        df_stat["SCR-R"] = score_right_list
        score_group_by = df_res["Y-SCR"].groupby(df_res["BIN"])
        label_group_by = df_res["LB"].groupby(df_res["BIN"])
        score_stat = score_group_by.mean()
        hit_stat = label_group_by.count()
        bad_stat = label_group_by.sum()
        df_stat["SCR-AVG"] = score_stat
        df_stat["HITS"] = hit_stat
        df_stat["BAD-HITS"] = bad_stat
        df_stat["ODDS"] = (df_stat["HITS"] - df_stat["BAD-HITS"]) / (df_stat["BAD-HITS"] + 0.0001)
        df_stat["OD-RA"] = (df_stat["BAD-HITS"] / (df_stat["HITS"]) + 0.0001) * 100
        df_stat.fillna(0, inplace=True)
        hit_total = sum(df_stat["HITS"])
        bad_total = sum(df_stat["BAD-HITS"])
        hit_acc = 0
        bad_acc = 0
        for i in range(stat_bins):
            hit_acc += df_stat["HITS"][i]
            bad_acc += df_stat["BAD-HITS"][i]
            df_stat["CUM-PA"][i] = (hit_acc / hit_total) * 100
            df_stat["CUM-OD"][i] = (bad_acc / bad_total) * 100
            df_stat["RE-OD"][i] = (bad_acc / (hit_acc + 0.0001)) * 100
        return df_stat

