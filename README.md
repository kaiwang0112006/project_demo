# 说明

关于sklearn2pmml更新后，有一个函数不能使用，出错信息:

	>>> from sklearn2pmml.feature_extraction.tabular import FeatureBinarizer
	Traceback (most recent call last):
	  File "<stdin>", line 1, in <module>
	ModuleNotFoundError: No module named 'sklearn2pmml.feature_extraction.tabular'

重写了一个特征生成的包，使用方式为:

    # feature engineer
    # data是一个dataframe
    standard_feature_obj = standard_feature_tree(data, 'target')
    standard_feature_obj.categ_continue_auto()
    standard_feature_obj.miss_inf_trans()
    standard_feature_obj.categ_label_trans()
    standard_feature_obj.apply_standardscale_classification()
    
    model = LogisticRegression(penalty='l2', C=1.0,class_weight='balanced',solver='newton-cg')
    
    X_train = standard_feature_obj.scaled_sample_x
    y_train = standard_feature_obj.sample_y
    
    model.fit(X_train, y_train)
    
*standard_feature_tree* 是特征生成的的功能类，*categ_continue_auto* 是识别类别特征和连续特征，*miss_inf_trans*是对连续特征处理缺失值，*categ_label_trans* 将类别特征转化为数值，*apply_standardscale_classification* 做归一化

# 更新 20171105

封装贝叶斯优化过程模仿sklearn的方式

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

    bayesopsObj = bayes_ops(estimator=LGBMClassifier, param_grid=parms, cv=10, intdeal=intdeal, middledeal=middledeal, 
                    maxdeal=maxdeal, 
                    score_func=make_scorer(score_func=accuracy_score, greater_is_better=True),
                    )
    bayesopsObj.run(X=X_train, Y=y_train)
    parms = bayesopsObj.baseparms
    model = LGBMClassifier(**parms)
    
# 更新 20171119

重构流水线作业的特征生产过程

    # 判断连续和类别型特征
    categoricalDomain, continuousDomain = categ_continue_auto_of_df(data,'Response')
    print(categoricalDomain, continuousDomain)
    
    # 串行流水作业
    step1 = ('infinite', InfClass(continuous=continuousDomain,method='max_min')) # 正负无穷大处理为最大最小值
    step2 = ("imputer", ImputerClass(continuous=continuousDomain,strategy='mean'))  # 连续变量缺失值处理
    step3 = ('onehot', OneHotClass(catego=categoricalDomain, miss='missing'))    ＃ 类别型变量独热编码

    pipeline = Pipeline(steps=[step1,step2,step3])
    newdata = pipeline.fit_transform(data)
    
处理前的数据：

	   Product_Info_2   Ins_Age     three     four Response
	a             inf  0.879701 -0.704034      bar     True
	b             NaN       NaN       NaN      foo      NaN
	c       -1.327915      -inf  1.416521      bar    False
	d             NaN       NaN       NaN  missing      NaN
	e        2.642622 -1.017065  0.324612      bar     True
	f        0.358531 -0.473738 -2.143172      bar     True
	g             NaN       NaN       NaN  missing      NaN
	h        0.431737 -0.502631 -0.352882      bar     True
	
处理后的数据：

	   Product_Info_2   Ins_Age     three Response  four_bar  four_foo
	a        2.642622  0.879701 -0.704034     True       1.0       0.0
	b        0.949519 -0.426159 -0.291791      NaN       0.0       1.0
	c       -1.327915 -1.017065  1.416521    False       1.0       0.0
	d        0.949519 -0.426159 -0.291791      NaN       0.0       0.0
	e        2.642622 -1.017065  0.324612     True       1.0       0.0
	
# 更新 20171206

增加label_encoder, 通过下面代码：

    categoricalDomain, continuousDomain = categ_continue_auto_of_df(data,'Response')
    print(categoricalDomain, continuousDomain)
    
    # 串行流水作业 version 1
    step1 = ('infinite', InfClass(continuous=continuousDomain,method='max_min')) # 正负无穷大处理为最大最小值
    step2 = ("imputer", ImputerClass(continuous=continuousDomain,strategy='mean'))  # 连续变量缺失值处理
    step3 = ('label_encode', label_encoder_sk(cols=categoricalDomain))

    pipeline = Pipeline(steps=[step1,step2,step3])
    newdata = pipeline.fit_transform(data)
    
将原数据：

	   Product_Info_2   Ins_Age     three     four Response
	a             inf -0.347696  0.512739      bar     True
	b             NaN       NaN       NaN      foo      NaN
	c        0.281222      -inf -0.935522      bar     True
	d             NaN       NaN       NaN  missing      NaN
	e        0.015555  3.388281  0.574353      bar     True
	f       -0.784892 -1.576177 -1.661712      bar    False
	g             NaN       NaN       NaN  missing      NaN
	h       -0.332099 -0.071977 -0.290078      bar    False
	
转化为：

	   Product_Info_2   Ins_Age     three  four Response
	a        0.281222 -0.347696  0.512739     1     True
	b       -0.107798 -0.036749 -0.360044     2      NaN
	c        0.281222 -1.576177 -0.935522     1     True
	d       -0.107798 -0.036749 -0.360044     3      NaN
	e        0.015555  3.388281  0.574353     1     True
	
加入最大最小值归一化：

    # data preprocess version 3
    # 判断连续和类别型特征
    categoricalDomain, continuousDomain = categ_continue_auto_of_df(data,'Response')
    #print(categoricalDomain, continuousDomain)

    # 串行流水作业 version 3minmaxScalerClass
    step1 = ('infinite', InfClass(continuous=continuousDomain,method='max_min')) # 正负无穷大处理为最大最小值
    step2 = ("imputer", ImputerClass(continuous=continuousDomain,strategy='mean'))  # 连续变量缺失值处理
    step3 = ('onehot', OneHotClass(catego=categoricalDomain, miss='missing')) # 类别特征独热编码
    step4 = ('MinMaxScaler', minmaxScalerClass(cols=[],target="Response"))  # 最大最小值归一化

    pipeline = Pipeline(steps=[step1,step2,step3,step4])
    newdata = pipeline.fit_transform(data)
    
加入iv值计算:

    df = datasets.load_breast_cancer()
    ivobj = iv_pandas()
    datadf = pd.DataFrame(df.data,columns=[str(i) for i in range(30)])
    datadf['target'] = df.target
    #print(datadf.head())
    x=datadf['1']

    woe, iv = ivobj.cal_woe_iv(datadf,['1','2'],'target',nsplit=10,event=1)
    print(iv) # iv结果是 {'1': 1.2674164699321071, '2': 2.9048936041315248}
    
# 更新 20180110

unix时间戳指定时区转换

    from project_demo.tools.timeParse import *
    
    tz_utc_0 = datetime.timezone(datetime.timedelta(hours=0))
    tz_utc_8 = datetime.timezone(datetime.timedelta(hours=8))
    print(unix2datetime(1503676813732,tz_utc_0,'s')) # 转换为0时区,单位秒
    print(unix2datetime(1503676813732,tz_utc_8,'ms')) # 转化为东八区，单位毫秒

sas时间转换函数的名称做调整

# 更新 20180128

添加mongodb的数据库连接对象，如下面用法

    from project_demo.tools.dbUtil import *

    connObj = MongoConnect('localhost', 27017)
    connObj.connect()
    connObj.authdb('cuishou', 'dbuser', 'readonly')
    
    textcol = connObj.db['text_less']   
    result = textcol.find() 

# 更新20180426

计算ks:

    from project_demo.tools.evaluate import *
    ksobj = ks_statistic(yprob=y_pred_prob, ytrue=y_valid)
    ksobj.cal_ks()
    print('train ks=%s' % str(ksobj.ks))
    ksobj.cal_ks_with_plot(plot="ks_test.png") #画ks曲线

修复iv计算的bug

psi计算:

    from project_demo.tools.evaluate import *
    psiobj = psi()
    p = psiobj.fit_mdl([1, 0, 1, 0, 1, 0, 0],box=False).cal([1, 0, 1, 0, 1, 1, 1])
    
# 更新20180530

自动算出每个入模变量的分箱，WOE,IV，用于检查变量的风险模式

	from project_demo.tools.evaluate import *
	import random
	vobj = varible_exam(list(range(100)),[random.choice([1, 0]) for i in range(100)]) # list(range(100)) 对应变量值， [random.choice([1, 0]) for i in range(100)]对应标签
	
	for key in vobj.binning(10).good_bad_rate().woe_iv().bindata:
	    data = vobj.binning(10).good_bad_rate().woe_iv().bindata[key]
	    if key == 'total':
	        data['range'] = 'total'
	    else:
	        data['range'] = "%s-%s" % (str(key[0]),str(key[1]))
	    print(data)

output:

	{'count': 10, 'good': 4, 'bad': 6, 'total': 10, 'total_ratio': 0.1, 'bad_odds': 0.6, '%good': 0.08888888888888889, '%bad': 0.10909090909090909, 'woe': 0.20479441264601306, 'iv': 0.004137260861535616, 'range': '0.0-9.9'}
	{'count': 10, 'good': 4, 'bad': 6, 'total': 10, 'total_ratio': 0.1, 'bad_odds': 0.6, '%good': 0.08888888888888889, '%bad': 0.10909090909090909, 'woe': 0.20479441264601306, 'iv': 0.004137260861535616, 'range': '9.9-19.8'}
	{'count': 10, 'good': 4, 'bad': 6, 'total': 10, 'total_ratio': 0.1, 'bad_odds': 0.6, '%good': 0.08888888888888889, '%bad': 0.10909090909090909, 'woe': 0.20479441264601306, 'iv': 0.004137260861535616, 'range': '19.8-29.700000000000003'}
	{'count': 10, 'good': 4, 'bad': 6, 'total': 10, 'total_ratio': 0.1, 'bad_odds': 0.6, '%good': 0.08888888888888889, '%bad': 0.10909090909090909, 'woe': 0.20479441264601306, 'iv': 0.004137260861535616, 'range': '29.700000000000003-39.6'}
	{'count': 10, 'good': 4, 'bad': 6, 'total': 10, 'total_ratio': 0.1, 'bad_odds': 0.6, '%good': 0.08888888888888889, '%bad': 0.10909090909090909, 'woe': 0.20479441264601306, 'iv': 0.004137260861535616, 'range': '39.6-49.5'}
	{'count': 10, 'good': 4, 'bad': 6, 'total': 10, 'total_ratio': 0.1, 'bad_odds': 0.6, '%good': 0.08888888888888889, '%bad': 0.10909090909090909, 'woe': 0.20479441264601306, 'iv': 0.004137260861535616, 'range': '49.5-59.400000000000006'}
	{'count': 10, 'good': 8, 'bad': 2, 'total': 10, 'total_ratio': 0.1, 'bad_odds': 0.2, '%good': 0.17777777777777778, '%bad': 0.03636363636363636, 'woe': -1.586965056582042, 'iv': 0.22441930093079385, 'range': '59.400000000000006-69.3'}
	{'count': 10, 'good': 5, 'bad': 5, 'total': 10, 'total_ratio': 0.1, 'bad_odds': 0.5, '%good': 0.1111111111111111, '%bad': 0.09090909090909091, 'woe': -0.20067069546215124, 'iv': 0.004053953443679821, 'range': '69.3-79.2'}
	{'count': 10, 'good': 2, 'bad': 8, 'total': 10, 'total_ratio': 0.1, 'bad_odds': 0.8, '%good': 0.044444444444444446, '%bad': 0.14545454545454545, 'woe': 1.1856236656577395, 'iv': 0.11975996622805447, 'range': '79.2-89.10000000000001'}
	{'count': 10, 'good': 5, 'bad': 4, 'total': 9, 'total_ratio': 0.09, 'bad_odds': 0.4444444444444444, '%good': 0.1111111111111111, '%bad': 0.07272727272727272, 'woe': -0.4238142467763609, 'iv': 0.016267617553032035, 'range': '89.10000000000001-99.0'}
	{'count': 100, 'good': 45, 'bad': 55, 'total': 100, 'total_ratio': 1.0, 'bad_odds': 0.55, '%good': 1.0, '%bad': 1.0, 'woe': 0.0, 'iv': 0.38932440332477386, 'range': 'total'}	    