#coding:utf-8
import json, sys
from datetime import datetime

from domain_schema_after_201706 import DOMAIN_SCHEMA

def convert(data, name, fschema):
    '''
    数据格式化转换
    '''
    # 如果没有记录返回缺失
    if not name in data:
        return fschema.get('default', None)
    data = data[name]
    handler = fschema['type']
    # 如果格式为缺失返回空
    if handler is None:
        return ''
    # 检查对象是否可被调用,可调用返回格式化后的数据
    elif callable(handler):
        return handler(data)
    # 日期格式进行处理
    elif handler == 'timestamp[ms]':
        return datetime.fromtimestamp(float(data) / 1000)
    elif handler == 'timestamp[s]':
        return datetime.fromtimestamp(float(data))
    elif handler == 'datetime[m]':
        try:
            return datetime.strptime(str(data), '%Y-%m-%d %H:%M')
        except:
            return None
    return None

def process(data, schema):
    '''
    区分读取的一条记录key与values,并将其分别拼接
    '''
    keys, values = [], []
    for name in schema['keys']:
        # 数据的格式
        field_schema = schema['fields'][name]
        keys.append(convert(data, name, field_schema))
    for name in schema['values']:
        field_schema = schema['fields'][name]
        try:
            values.append(convert(data, name, field_schema))
        except:
            values.append(None)
            # print(name, data[name], data, field_schema)
            # exit(1)
    return keys, values

def output_key(x):
    if type(x) == datetime:
        return str(x.timestamp())
    return str(x)

def output_value(x):
    if type(x) == datetime:
        return x.timestamp()
    return x

# 判断如果在此文件中执行，如果被其他文件导入，不执行
if __name__ == '__main__':
    # 遍历输入内容的每行
    for line in sys.stdin:
        data = None
        # 异常处理
        try:
            data = json.loads(line)
        except:
            continue

        # 只拿出需要衍生的记录
        if type(data) != dict or not 'mid' in data or not data['mid'] in DOMAIN_SCHEMA:
            continue
        ## hack here: CON01 ==> IDCARD01
        # 因为身份采集信息ios端部分数据上报mid为con01故做此处理
        if data['mid'] == 'CON01' and not 'phone' in data and 'id_card_number' in data:
            data['mid'] = 'IDCARD01'

	#lvjinwei:修改call_type替换为type，解决5月数据提取异常问题
        if data['mid'] == 'CAL01' and 'call_type' in data.keys():
            typevalue = data['call_type']
            data.pop('call_type')
            data['type'] = data.get('type', typevalue)

        domain = data['mid']
        # 根据字典拿出key,value值
        keys, values = process(data, DOMAIN_SCHEMA[domain])
        # 将value拆分
        for i in range(len(values)):
            values[i] = output_value(values[i])   

        ## hack here: expand numbers
        # 通讯录多个号码拆分成多条记录
        if data['mid'] == 'CON01' and ',' in data['phone']:
        #    idx_masked = DOMAIN_SCHEMA[domain]['values'].index('phone_masked')
            idx = DOMAIN_SCHEMA[domain]['values'].index('phone')
            try:
                phones = values[idx].split(',')
            except:
                print(data)
            # phones = values[idx].split(',')
            new_values = []
            for i in range(len(phones)):
                v = values[:]
                v[idx] = phones[i]
                #v[idx] = phones[i]
                new_values.append(v)
            values = new_values
        # 输出为一堆key,一个value的形式
        print('\t'.join(map(output_key, keys)), end='\t')
        print(json.dumps(values, ensure_ascii=False, separators=(',',':')))
