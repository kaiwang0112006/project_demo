# -*-coding: utf-8-*-
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def gbm_format(parms):
    intdeal = ['max_bin','max_depth','max_drop','min_child_samples',
               'min_child_weight','n_estimators','num_leaves','scale_pos_weight',
               'subsample_for_bin','subsample_freq'] # int类参数
    middledeal = ['colsample_bytree','drop_rate','learning_rate',
                  'min_split_gain','skip_drop','subsample',''] # float， 只能在0，1之间
    maxdeal = ['reg_alpha','reg_lambda','sigmoid']  # float，且可以大于1
    for k in parms:
        if k in intdeal:
            parms[k] = int(parms[k])
        elif k in maxdeal:
            parms[k] = max(parms[k], 0)
        elif k in middledeal:
            parms[k] = max(min(parms[k], 1), 0)
    return parms
        

def gbmr_eval_bak(boosting_type='gbdt', colsample_bytree=0.46539839899078977,drop_rate=0.1, is_unbalance=True,learning_rate=0.083660800960886683, max_bin=14, max_depth=7,max_drop=50, min_child_samples=70, min_child_weight=1,min_split_gain=0.024313726655626314, n_estimators=581, nthread=-1,num_leaves=128, objective='binary', reg_alpha=0, reg_lambda=1,scale_pos_weight=1, seed=27, sigmoid=1.0, silent=True,skip_drop=0.5, subsample=1, subsample_for_bin=50000,subsample_freq=3, uniform_drop=False, xgboost_dart_mode=False):
    gbmr = LGBMClassifier(
        boosting_type='gbdt', 
        colsample_bytree=max(min(colsample_bytree, 1), 0),
        drop_rate=drop_rate, 
        is_unbalance=True,
        learning_rate=0.083660800960886683, max_bin=14, max_depth=7,
        max_drop=50, min_child_samples=70, min_child_weight=1,
        min_split_gain=0.024313726655626314, n_estimators=581, nthread=-1,
        num_leaves=128, objective='binary', reg_alpha=0, reg_lambda=1,
        scale_pos_weight=1, seed=27, sigmoid=1.0, silent=True,
        skip_drop=0.5, subsample=1, subsample_for_bin=50000,
        subsample_freq=3, uniform_drop=False, xgboost_dart_mode=False
    )
    
    score =  cross_val_score(gbmr, X=X, y=y, scoring=make_scorer(score_func=accuracy_score, greater_is_better=True), cv=5, verbose=0, pre_dispatch=1)
    return np.array(score).mean()

 

