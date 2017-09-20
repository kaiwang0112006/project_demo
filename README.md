# 说明

关于sklearn2pmml更新后，有一个函数不能使用，重写了一个特征生成的包，使用方式为:

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