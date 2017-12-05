#coding:utf-8
import sys

zuid_dict = {}

if __name__ == '__main__':
    # 生成ugid与zuid对应关系字典
    mfile = open('../data_raw/ugid_zuid.201701_201710')
    for line in mfile:
        ugid, zuid = line.rstrip('\r\n').split(',', 1)
        if zuid in zuid_dict:
            zuid_dict[zuid].append(ugid)
        else:
            zuid_dict[zuid] = [ ugid ]
    # 将输入的数据ugid缺失的通过zuid对应
    for line in sys.stdin:
        line = line.rstrip('\r\n')
        try:
            ugid, zuid, data = line.split('\t', 2)
        except:
            continue
        # print(ugid)
        # print(zuid)
        # print(data)
        if ugid:
            print(line)
            continue
        if not zuid:
            continue
        try:
            for i in zuid_dict[zuid]:
                print('\t'.join((i, zuid, data)))
        except:
            continue
        
