# -*- coding:utf-8 -*-
"""
@author: kaiwang
@time: 2019/06/13
"""
from .EqualWidthBinning import EqualWidthBinning
import dill

class prob_scoring(object):
    def __init__(self, list_prob, list_label, bins=20, standard_score=500, standard_odds=40, pdo=20):
        self.list_prob = list_prob
        self.list_label = list_label
        self.bins = bins
        self.standard_score = standard_score
        self.standard_odds = standard_odds
        self.pdo = pdo

    def output_binning(self, fileout=''):
        binning = EqualWidthBinning(list_p=self.list_prob, list_label=self.list_label, bins=self.bins, standard_score=self.standard_score, standard_odds=self.standard_odds, pdo=self.pdo)
        self.__df_mapping = binning.get_mapping_df()
        print(fileout)
        dill.dump(self.__df_mapping.to_dict(), open(fileout, "wb"))

