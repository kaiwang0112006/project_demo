# 此代码用来从原始json文件中抽取数据
# 原始数据量太大,所以在sampleRun中省略了这个步骤
# 输入1.当月起始日期,2.当月结束日期,3.当月月份
SRC_PATH="../data_raw"
# 生成$1 到 $2的数字序列
DAYS=`seq $1 $2`
# month=$3
for d in $DAYS; do
    # 输出当前日期与数据日期
    echo `date` $d;
    # 将log文件压缩文件内容传给convert.py
    zcat $SRC_PATH/sea.*.gz | grep 'ACQ0[1-3]\|IDCARD01\|CAL01\|CON01\|DEV01' \
        | /usr/bin/python3.5 convert_after_201706.py | LC_ALL=C uniq \
        | /usr/bin/python3.5 ugid_mapping_after_201706.py | LC_ALL=C sort -u -S2G \
        | /usr/bin/python3.5 merge_after_201706.py \
	> ../data_raw/data.$d.log
done
