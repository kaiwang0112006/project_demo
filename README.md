# ˵��

����sklearn2pmml���º���һ����������ʹ�ã���д��һ���������ɵİ���ʹ�÷�ʽΪ:

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

# ����

��װ��Ҷ˹�Ż�����

    # model ops
    bayesopsObj = bayes_ops(X=X_train, Y=y_train, estimator=LGBMClassifier)
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
    bayesopsObj.run(parms=parms, cv=10, intdeal=intdeal, middledeal=middledeal, 
                    maxdeal=maxdeal, 
                    score_func=make_scorer(score_func=accuracy_score, greater_is_better=True),
                    )
    
    
    parms = bayesopsObj.baseparms
    model = LGBMClassifier(**parms)