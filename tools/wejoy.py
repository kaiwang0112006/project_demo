# -*- coding: utf8 -*-
import pandas as pd

def feature_available(data, avi=0.9, frq=0.1, ct=-1, fadd=['overdue_day', 'created', 'loan_id']):
    data_feature = pd.DataFrame(data.count(), columns=['cnt'])
    data_feature.reset_index(level=0, inplace=True)
    data_feature_filter = data_feature[data_feature['cnt'] >= (len(data) * avi)]

    if ct==-1:
        ct= len(data)*frq
    features = []
    for f in data.columns:
        try:
            int(f)
            if f in list(data_feature_filter['index']) and len(set(data[f]))>ct:
                features.append(f)
        except:
            # traceback.print_exc()
            pass

    for f in fadd:
        features.append(f)
    return data_feature, features

