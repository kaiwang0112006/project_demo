# -*-coding: utf-8-*-
import pandas as pd
from tools.features_engine import *
from tools.evaluate import *

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

def main():
    topicdata = pd.read_csv(r'/project/wk/tb/20170906/user_out.csv')
    tbjxldata = pd.read_csv(r'/project/wk/tb/data/tbjxl_r360dtl_data20170713_uft8.csv')
    tbjxldata = tbjxldata[(tbjxldata['target']!=2) & tbjxldata['flg_sample']==1] # 有卡

    topicdata.rename(columns={"ugid": "user_gid"},inplace=True)
    rawdata = pd.merge(topicdata,tbjxldata,on='user_gid')

    exclude = ['cust_nm', 'register_mobile', 'flg_jxl', 'flg_tb', 'flg_sample', 'user_gid', 'IDCardNO', 
               'decision_tm', 'usertype','ugid','weight','phone','id_card_1','mobile_auth','first_decision_tm','register_time','credit_history',
               'cust_nm_sha','id_card_sha','mobile_sha','cust_nm_1','target1','cust_perf','source']
    
    features = [f for f in rawdata.columns if f not in exclude]
    data = rawdata[features]
    data = data.replace('@', np.nan)
    data = data.replace(-9999976, np.nan)
    data = data.replace(-99999976, np.nan)
    data = data.replace(-9999977, np.nan)
    data = data.replace(-9999978, np.nan)
    data = data.replace(-99999980, np.nan)
    data = data.replace(-99998.0, np.nan)
    print("data shape %s" % str(data.shape))
    
    # count missing data in each column
    invest = data.isnull().sum()
    for i in invest.index:
        if invest[i] > 0:
            break
            print("feature %s have missing %s data" % (i,str(invest[i])))
    
    # feature engineer
    standard_feature_obj = standard_feature_kexin(data, 'target')
    standard_feature_obj.categ_continue_auto()
    standard_feature_obj.miss_inf_trans()
    standard_feature_obj.categ_trans()
    standard_feature_obj.apply_standardscale_classification()
    
    # logistic
    # model = LogisticRegression(penalty='l2', C=1.0,class_weight='balanced',solver='newton-cg')
    model = LGBMClassifier(boosting_type='gbdt', colsample_bytree=0.55020108411564301, is_unbalance=True,
            learning_rate=0.05, max_bin=13, max_depth=4,
            max_drop=50, min_child_samples=23, min_child_weight=2,
            min_split_gain=0.074005289325428367, n_estimators=193, nthread=-1,
            num_leaves=128, objective='binary', reg_alpha=0, reg_lambda=1,
            scale_pos_weight=1, seed=27, sigmoid=1.0, silent=True,
            skip_drop=0.5, subsample=1, subsample_for_bin=50000,
            subsample_freq=5, uniform_drop=False, xgboost_dart_mode=False)
    
    
    X_train = standard_feature_obj.scaled_sample_x
    y_train = standard_feature_obj.sample_y
    
    model.fit(X_train, y_train)
    
    # trainingset evaluation
    print('trainingset evaluation')
    y_pred = model.predict(X_train)
    y_pred_prob = model.predict_proba(X_train)[:,0]
    acc = accuracy_score(y_pred, y_train, normalize=True)
    print('acc=%s' % str(acc))
    auc = roc_auc_score(y_score=y_pred, y_true=y_train.values)
    print('auc=%s' % str(auc))
    #evl.ks_curve(Y_true = y_train.values, Y_predprob = y_pred_prob, fig_path = 'lgr_train.png')
    ksobj = ks_statistic(yprob=y_pred_prob, ytrue=y_train.values)
    ksobj.cal_ks()
    print('ks=%s' % str(ksobj.ks))
    
    # testset evaluation
    print('testset evaluation')
    X_test = standard_feature_obj.scaled_test_x
    y_test = standard_feature_obj.test_y
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,0]
    acc = accuracy_score(y_pred, y_test, normalize=True)
    print('acc=%s' % str(acc))
    auc = roc_auc_score(y_score=y_pred, y_true=y_test.values)
    print('auc=%s' % str(auc))
    #evl.ks_curve(Y_true = y_train.values, Y_predprob = y_pred_prob, fig_path = 'lgr_train.png')
    ksobj = ks_statistic(yprob=y_pred_prob, ytrue=y_test.values)
    ksobj.cal_ks()
    print('ks=%s' % str(ksobj.ks))    
    
if __name__ == '__main__':
    main()