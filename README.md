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
 
# 更新20180522

更新iv计算：

按变量n等分做分箱:

    from project_demo.tools.evaluate import * 
    woe, iv = ivobj.cal_woe_iv_by_x(datadf,['v1','v2'],'target',nsplit=10,event=1)
    
按y做n等分做分箱:

    from project_demo.tools.evaluate import * 
    woe, iv = ivobj.cal_woe_iv_by_y(datadf,['v1','v2'],'target',nsplit=10,event=1)