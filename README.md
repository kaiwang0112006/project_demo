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