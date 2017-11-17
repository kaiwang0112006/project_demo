# -*-coding: utf-8-*-
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
import numpy as np
def parm_format(parms, intdeal, middledeal, maxdeal):
    '''
         整理模型参数的格式，intdeal是int类参数的列表，middledeal是小数类参数，maxdeal是可大与1的整数类参数
         如下面是lightgbm的格式：
#     intdeal = ['max_bin','max_depth','max_drop','min_child_samples',
#                'min_child_weight','n_estimators','num_leaves','scale_pos_weight',
#                'subsample_for_bin','subsample_freq'] # int类参数
#     middledeal = ['colsample_bytree','drop_rate','learning_rate',
#                   'min_split_gain','skip_drop','subsample',''] # float， 只能在0，1之间
#     maxdeal = ['reg_alpha','reg_lambda','sigmoid']  # float，且可以大于1
    '''
    for k in parms:
        if k in intdeal:
            parms[k] = int(parms[k])
        elif k in maxdeal:
            parms[k] = max(parms[k], 0)
        elif k in middledeal:
            parms[k] = max(min(parms[k], 1), 0)
    return parms

class bayes_ops(object):
    '''
    object that implement Bayesian Optimization using BayesianOptimization
    (https://github.com/fmfn/BayesianOptimization)
    '''
    def __init__(self, estimator, param_grid, cv, intdeal, middledeal, maxdeal, score_func,baseparms={}, num_iter=100, init_points=15, n_iter=25, acq='ucb', kappa=2.576, xi=0.0, gp_params={"alpha": 1e-5, "n_restarts_optimizer": 2}):
        '''
        estimator need to have fit function
        '''
        self.estimator = estimator
        self.baseparms = baseparms
        self.parms = param_grid
        self.cv = cv
        self.intdeal = intdeal
        self.middledeal = middledeal
        self.maxdeal = maxdeal
        self.score_func = score_func
        self.num_iter = num_iter
        self.init_points = init_points
        self.acq = acq 
        self.kappa = kappa 
        self.xi = xi
        self.gp_params = gp_params
        

    def _est_eval(self, **parms):
        parms = parm_format(parms, self.intdeal, self.middledeal, self.maxdeal)
        for p in self.baseparms:
            parms[f] = self.baseparms[p]
        estmr = self.estimator(**parms)
        score =  cross_val_score(estmr, X=self.X, y=self.Y, scoring=self.score_func, cv=self.cv, verbose=0, pre_dispatch=1)
        return np.array(score).mean()  

    def run(self, X,Y):
        self.X = X
        self.Y = Y
        estmrBO = BayesianOptimization(self._est_eval, 
                                      self.parms
                                     )
        
        estmrBO.maximize(init_points=self.init_points, n_iter=self.num_iter, acq=self.acq, kappa=self.kappa, xi=self.xi,**self.gp_params)
        self.baseparms = parm_format(estmrBO.res['max']['max_params'], self.intdeal, self.middledeal, self.maxdeal)

     
