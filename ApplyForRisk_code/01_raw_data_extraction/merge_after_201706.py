#coding:utf-8
import itertools, json, operator, sys
from domain_schema_after_201706 import DOMAIN_SCHEMA

def input_keys(line):
    return line.split('\t')[:-2]

def input_value_getter(line):
    '''
    取出最后一个元素即values
    '''
    p = line.rfind('\t')
    values = json.loads(line[p+1:])
    return values if type(values[0]) == list else [values]

def process(key, values):
    domain = key[-1]
    if not domain in DOMAIN_SCHEMA:
        return
    schema = DOMAIN_SCHEMA[domain]

    dedup_set = None
    if 'dedup' in schema:
        dedup_idx = [ schema['values'].index(k) for k in schema['dedup'] ]
        dedup_getter = operator.itemgetter(*dedup_idx)
        dedup_set = set()

    new_values = []
    for vs in values:
        for v in vs:
            if not dedup_set is None:
                k = dedup_getter(v)
                if k in dedup_set:
                    continue
                dedup_set.add(k)
            new_values.append(v)

    print('\t'.join(key), end='\t\t')
    print(json.dumps(new_values, ensure_ascii=False, separators=(',',':')))

if __name__ == '__main__':
    # 将纵向合并的文件转为以key为主键横向合并
    for key, values in itertools.groupby(sys.stdin, input_keys):
        process(key, map(input_value_getter, values))
