
# coding: utf-8

# In[1]:
import pandas as pd
#import json
#import numpy as np
import sys


# In[2]:
mid_name=('ACQ01','ACQ02','ACQ03','IDCARD01','$LABEL','BCC01','DEV01','CAL01','CON01','LOGIN01')


# In[3]:
#if __name__ == '__main__':
input_str = sys.argv[1]
    
#input_str = '201612'

# In[5]:
print('开始读文件')
# 筛选四种数据
data = []
with open('../data_raw/data.%s.log'%input_str, 'r') as f:
    for line in f:
        try:
            tmp_list = line.split('\t')
            if tmp_list[3] in mid_name: 
                data.append(line.split('\t'))
        except:
            print(line)
            continue
            


# In[ ]:
print('文件读取结束,开始读取名单')
# 读取名单overdue_status = pd.read_csv('./labelfile_edited.txt')
reader = pd.read_csv(r"../data_raw/APP_5K_20171120_ADD_UGID.csv")
#读出数据,ignore_index各文件块数据是否重新赋值索引,返回一个dataframe
# acq01 = pd.concat(reader, ignore_index=False)
reader = pd.DataFrame(reader['ugid']).drop_duplicates()


# In[ ]:

pd1 = pd.DataFrame(data)
pd1.columns = ['ugid', 'zuid', 'appid', 'mid', 'unknwon', 'values']


# In[ ]:
print('读取结束,开始拼接')
# 数据拼接筛选
pd2 = pd.merge(reader,pd1,how='left',on='ugid')
pd3 = pd2[pd2.mid == pd2.mid].reset_index(drop=True)


# In[ ]:
print('开始输出文件')
# 输出数据
pd3.to_csv('../data_mid/data.%s.screen_all.log'%input_str, index = False,  sep='\t' ,header=False)

