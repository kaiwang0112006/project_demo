# ˵��

����sklearn2pmml���º���һ����������ʹ�ã�������Ϣ:

	>>> from sklearn2pmml.feature_extraction.tabular import FeatureBinarizer
	Traceback (most recent call last):
	  File "<stdin>", line 1, in <module>
	ModuleNotFoundError: No module named 'sklearn2pmml.feature_extraction.tabular'

��д��һ���������ɵİ���ʹ�÷�ʽΪ:

    # feature engineer
    # data��һ��dataframe
    standard_feature_obj = standard_feature_tree(data, 'target')
    standard_feature_obj.categ_continue_auto()
    standard_feature_obj.miss_inf_trans()
    standard_feature_obj.categ_label_trans()
    standard_feature_obj.apply_standardscale_classification()
    
    model = LogisticRegression(penalty='l2', C=1.0,class_weight='balanced',solver='newton-cg')
    
    X_train = standard_feature_obj.scaled_sample_x
    y_train = standard_feature_obj.sample_y
    
    model.fit(X_train, y_train)
    
*standard_feature_tree* ���������ɵĵĹ����࣬*categ_continue_auto* ��ʶ���������������������*miss_inf_trans*�Ƕ�������������ȱʧֵ��*categ_label_trans* ���������ת��Ϊ��ֵ��*apply_standardscale_classification* ����һ��

# ���� 20171105

��װ��Ҷ˹�Ż�����ģ��sklearn�ķ�ʽ

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
    # ���������ʽ����ʵֻ��Ҫ�ṩparms��Ĳ�������
    intdeal = ['max_bin','max_depth','max_drop','min_child_samples',
               'min_child_weight','n_estimators','num_leaves','scale_pos_weight',
               'subsample_for_bin','subsample_freq'] # int�����
    middledeal = ['colsample_bytree','drop_rate','learning_rate',
                  'min_split_gain','skip_drop','subsample',''] # float�� ֻ����0��1֮��
    maxdeal = ['reg_alpha','reg_lambda','sigmoid']  # float���ҿ��Դ���1

    bayesopsObj = bayes_ops(estimator=LGBMClassifier, param_grid=parms, cv=10, intdeal=intdeal, middledeal=middledeal, 
                    maxdeal=maxdeal, 
                    score_func=make_scorer(score_func=accuracy_score, greater_is_better=True),
                    )
    bayesopsObj.run(X=X_train, Y=y_train)
    parms = bayesopsObj.baseparms
    model = LGBMClassifier(**parms)
    
# ���� 20171119

�ع���ˮ����ҵ��������������

    # �ж����������������
    categoricalDomain, continuousDomain = categ_continue_auto_of_df(data,'Response')
    print(categoricalDomain, continuousDomain)
    
    # ������ˮ��ҵ
    step1 = ('infinite', InfClass(continuous=continuousDomain,method='max_min')) # �����������Ϊ�����Сֵ
    step2 = ("imputer", ImputerClass(continuous=continuousDomain,strategy='mean'))  # ��������ȱʧֵ����
    step3 = ('onehot', OneHotClass(catego=categoricalDomain, miss='missing'))    �� ����ͱ������ȱ���

    pipeline = Pipeline(steps=[step1,step2,step3])
    newdata = pipeline.fit_transform(data)
    
����ǰ�����ݣ�

	   Product_Info_2   Ins_Age     three     four Response
	a             inf  0.879701 -0.704034      bar     True
	b             NaN       NaN       NaN      foo      NaN
	c       -1.327915      -inf  1.416521      bar    False
	d             NaN       NaN       NaN  missing      NaN
	e        2.642622 -1.017065  0.324612      bar     True
	f        0.358531 -0.473738 -2.143172      bar     True
	g             NaN       NaN       NaN  missing      NaN
	h        0.431737 -0.502631 -0.352882      bar     True
	
���������ݣ�

	   Product_Info_2   Ins_Age     three Response  four_bar  four_foo
	a        2.642622  0.879701 -0.704034     True       1.0       0.0
	b        0.949519 -0.426159 -0.291791      NaN       0.0       1.0
	c       -1.327915 -1.017065  1.416521    False       1.0       0.0
	d        0.949519 -0.426159 -0.291791      NaN       0.0       0.0
	e        2.642622 -1.017065  0.324612     True       1.0       0.0
	
# ���� 20171206

����label_encoder, ͨ��������룺

    categoricalDomain, continuousDomain = categ_continue_auto_of_df(data,'Response')
    print(categoricalDomain, continuousDomain)
    
    # ������ˮ��ҵ version 1
    step1 = ('infinite', InfClass(continuous=continuousDomain,method='max_min')) # �����������Ϊ�����Сֵ
    step2 = ("imputer", ImputerClass(continuous=continuousDomain,strategy='mean'))  # ��������ȱʧֵ����
    step3 = ('label_encode', label_encoder_sk(cols=categoricalDomain))

    pipeline = Pipeline(steps=[step1,step2,step3])
    newdata = pipeline.fit_transform(data)
    
��ԭ���ݣ�

	   Product_Info_2   Ins_Age     three     four Response
	a             inf -0.347696  0.512739      bar     True
	b             NaN       NaN       NaN      foo      NaN
	c        0.281222      -inf -0.935522      bar     True
	d             NaN       NaN       NaN  missing      NaN
	e        0.015555  3.388281  0.574353      bar     True
	f       -0.784892 -1.576177 -1.661712      bar    False
	g             NaN       NaN       NaN  missing      NaN
	h       -0.332099 -0.071977 -0.290078      bar    False
	
ת��Ϊ��

	   Product_Info_2   Ins_Age     three  four Response
	a        0.281222 -0.347696  0.512739     1     True
	b       -0.107798 -0.036749 -0.360044     2      NaN
	c        0.281222 -1.576177 -0.935522     1     True
	d       -0.107798 -0.036749 -0.360044     3      NaN
	e        0.015555  3.388281  0.574353     1     True
	
���������Сֵ��һ����

    # data preprocess version 3
    # �ж����������������
    categoricalDomain, continuousDomain = categ_continue_auto_of_df(data,'Response')
    #print(categoricalDomain, continuousDomain)

    # ������ˮ��ҵ version 3minmaxScalerClass
    step1 = ('infinite', InfClass(continuous=continuousDomain,method='max_min')) # �����������Ϊ�����Сֵ
    step2 = ("imputer", ImputerClass(continuous=continuousDomain,strategy='mean'))  # ��������ȱʧֵ����
    step3 = ('onehot', OneHotClass(catego=categoricalDomain, miss='missing'))
    step4 = ('MinMaxScaler', minmaxScalerClass(cols=[],target="Response"))

    pipeline = Pipeline(steps=[step1,step2,step3,step4])
    newdata = pipeline.fit_transform(data)
