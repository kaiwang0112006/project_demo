# -*-coding: utf-8-*-
import pandas as pd
from tools.features_engine import *
from tools.evaluate import *
from tools.optimize import *

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from bayes_opt import BayesianOptimization

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def main():
    topicdata = pd.read_csv(r'/data/work/wk/tb/20170906/user_out.csv')
    tbjxldata = pd.read_csv(r'/data/work/wk/tb/data/tbjxl_r360dtl_data20170713_uft8.csv')
    tbjxldata = tbjxldata[(tbjxldata['target']!=2) & tbjxldata['flg_sample']==1] # 閺堝宕�

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
    standard_feature_obj = standard_feature_tree(data, 'target')
    standard_feature_obj.categ_continue_auto()
    standard_feature_obj.miss_inf_trans()
    standard_feature_obj.categ_label_trans()
    standard_feature_obj.format_train_test()
    #standard_feature_obj.apply_standardscale_classification()
    X_train = standard_feature_obj.sample_x
    y_train = standard_feature_obj.sample_y
    # model ops
    
    parms =  {
                #'x_train':X_train,
                #'y_train':y_train,
                'num_leaves': (15, 500),
                'colsample_bytree': (0.1, 1),
                'drop_rate': (0.1,1),
                'learning_rate': (0.001,0.05),
                'max_bin':(10,100),
                'max_depth':(2,20),
                'min_split_gain':(0.2,0.9),
                'min_child_samples':(10,200),
                'n_estimators':(100,3000),
                'reg_alpha':(0.1,100),
                'reg_lambda':(0.1,100),
                'sigmoid':(0.5,1),
                'subsample':(0.1,1),
                'subsample_for_bin':(10000,50000),
                'subsample_freq':(1,5)
              }
    # 参数整理格式，其实只需要提供parms里的参数即可
    intdeal = ['max_bin','max_depth','max_drop','min_child_samples',
               'min_child_weight','n_estimators','num_leaves','scale_pos_weight',
               'subsample_for_bin','subsample_freq'] # int类参数
    middledeal = ['colsample_bytree','drop_rate','learning_rate',
                  'min_split_gain','skip_drop','subsample',''] # float， 只能在0，1之间
    maxdeal = ['reg_alpha','reg_lambda','sigmoid']  # float，且可以大于1
    #bayesopsObj = bayes_ops(X=X_train, Y=y_train, estimator=LGBMClassifier)
    bayesopsObj = bayes_ops(estimator=LGBMClassifier, param_grid=parms, cv=10, intdeal=intdeal, middledeal=middledeal, 
                    maxdeal=maxdeal, 
                    score_func=make_scorer(score_func=accuracy_score, greater_is_better=True),
                    )
    bayesopsObj.run(X=X_train, Y=y_train)
    parms = bayesopsObj.baseparms
    model = LGBMClassifier(**parms)
    print(model)
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
    X_test = standard_feature_obj.test_x
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