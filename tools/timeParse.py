# -*- coding: utf8 -*-

import datetime 
import time 
import traceback
import calendar


MONTH = {'JAN':1, "FEB":2, "MAR":3, "APR":4, "MAY":5, "JUN":6, "JUL":7,"AUG":8, "SEP":9, "OCT":10, "NOV":11, "DEC":12}


def get_month_short_map_num():
    return {v: k for k,v in enumerate(calendar.month_abbr)}

def get_month_daylist(year,month,formats="%Y%m%d"):
	num_days = calendar.monthrange(year, month)[1]
	start = datetime.date(year=year,month=month,day=1)
	end = datetime.date(year=year,month=month,day=num_days)
	return get_day_list(start,end,formats)
	

def getOneDayDelta():

	'''

		DESC: 获取一天的datedelta量

	'''

	return datetime.datetime(2010,7,2)-datetime.datetime(2010,7,1)

	

def getDateDiffInDay(startDate,endDate):

	'''

		DESC: 获取两个日期之间相隔的天数

		PARAM: 

			startDate: datetime类型，起始日期

			endDate: datetime类型，终止日期

		RETURN:

			int

			e.g. 输入的日期为2010-07-01和2010-07-02的话，则返回1; 输入的日期为2010-07-02和2010-07-01的话，则返回-1

		AUTHOR: he_jian

		SINCE: 2010-08-05

	'''

	return (time.mktime(endDate.utctimetuple())-time.mktime(startDate.utctimetuple()))/86400



def get_month_list(startDate,endDate,format='%Y%m'):

	'''

		DESC: 获取两个日期之间间隔的月份列表

		PARAM: 

			startDate: datetime类型，起始日期

			endDate: datetime类型，终止日期

			format: 输出日期格式，如: '%Y%m','%Y-%m'

		RETURN:

			list

			e.g. startDate为2010-07-28,endDate为2010-10-01，format='%Y%m'，则返回值为：

				['201007','201008','201009','201010']

	'''

	step=datetime.datetime(2010,7,2)-datetime.datetime(2010,7,1)

	thedate=startDate

	retList = []

	while thedate < (endDate + step):

		themonthStr = datetime.datetime.strftime(thedate,format)

		if themonthStr not in retList:

			retList.append(themonthStr)

		thedate = thedate + step

	return retList

def get_day_list(startDate,endDate,format="%Y%m%d"):

	'''

		DESC: 获取两个日期之间间隔的日列表

		PARAM: 

			startDate: datetime类型，起始日期

			endDate: datetime类型，终止日期

			format: 输出日期格式，如: '%Y%m%d','%Y-%m-%d'

		RETURN:

			list

			e.g. startDate为2010-07-28,endDate为2010-08-02，format='%Y%m'，则返回值为：

				['20100728','20100729','20100730','20100731','20100801','20170802']

	'''

	step=datetime.datetime(2010,7,2)-datetime.datetime(2010,7,1)

	thedate=startDate

	retList = []

	while thedate < (endDate + step):

		theDayStr = datetime.datetime.strftime(thedate,format)

		if theDayStr not in retList:

			retList.append(theDayStr)

		thedate = thedate + step

	return retList

def sastime2standarddate(t):
	try:
	    return datetime.date(1960,1,1)+datetime.timedelta(days=int(float(t)))
	except:
		#traceback.print_exc()
		return 'NA' 

def sastime2standarddatetime(t):
    '''
    input as "20924"
    '''
    try:
        return datetime.datetime(1960,1,1)+datetime.timedelta(days=int(float(t)))
    except:
        #traceback.print_exc()
        return 'NA'
    
def sasdt2standarddatetime(t):
    '''
    input as "10JUL2017"
    '''
    day = int(t[:2])
    month = MONTH[t[2:5]]
    year = int(t[5:9])
    
    return datetime.datetime(year=year,month=month,day=day)

def unix2datetime_auto(t,tz):
    try:
        t = int(t)
    except:
        return "format error"
    
    if t>1e12:
        s = t/1000
    else:
        s = t

    return datetime.datetime.fromtimestamp(s,tz=tz)


def unix2datetime(t,tz,unit="s"):
    '''
    unit 可以是s和ms
    '''
    try:
        t = int(t)
    except:
        return "format error"

    if unit=='s':
        return datetime.datetime.fromtimestamp(t,tz)
    else:
        return datetime.datetime.fromtimestamp(t/1000,tz)

def string2datetime(t,format):
    '''
    t: "27-07-2017"  
    format: "%d-%m-%Y"
    return datetime
    '''
    return datetime.datetime.strptime(t,format)

def timeformat_Transform(t, format_in, format_out):
    t_ = datetime.datetime.strptime(t,format_in)
    return t_.strftime(format_out)

##  测试代码 

if __name__ == '__main__':
    t = timeformat_Transform('2019/5/24','%Y/%m/%d','%Y-%m-%d')
    print(t)
    print(type(t))

    '''
    print ('--------  TEST FOR getDateDiffInDay() -----------------------------')
    
    startDate = datetime.datetime.strptime('2010-07-01','%Y-%m-%d')
    
    endDate = datetime.datetime.strptime('2010-07-01','%Y-%m-%d')
    
    print ('startDate=%s,endDate=%s'%(startDate,endDate))
    
    diff = getDateDiffInDay(startDate,endDate)
    
    print ('diff=%d'%diff)
    
    print ('--------  TEST FOR get_month_list() ----------------------------')
    
    startDate = datetime.datetime.strptime('2010-04-02','%Y-%m-%d')
    
    endDate = datetime.datetime.strptime('2010-07-28','%Y-%m-%d')
    
    print ('startDate=%s,endDate=%s'%(startDate,endDate))
    
    retList = get_month_list(startDate, endDate, '%Y%m')
    
    print ('retList=%s'%retList)
    
    
    print ('--------  TEST FOR get_day_list() ----------------------------')
    
    startDate = datetime.datetime.strptime('2017-07-28','%Y-%m-%d')
    
    endDate = datetime.datetime.strptime('2016-08-02','%Y-%m-%d')
    
    print ('startDate=%s,endDate=%s'%(startDate,endDate))
    
    retList = get_day_list(startDate, endDate)
    
    print ('retList=%s'%retList)
    
    print ('--------  TEST FOR get_day_list() ----------------------------')
    print(sastime2standarddatetime(20945))
    
    print ('--------  TEST FOR get_month_daylist() ----------------------------')
    print(get_month_daylist(2017,4))
    
    print('--------  TEST FOR unix2datetime ----------------------------')
    tz_utc_0 = datetime.timezone(datetime.timedelta(hours=0))
    tz_utc_8 = datetime.timezone(datetime.timedelta(hours=8))
    print(unix2datetime(1503676813732,tz=tz_utc_0,unit='ms'))
    print(unix2datetime(1503676813732,tz=tz_utc_8,unit='ms'))
    
    print('--------  TEST FOR string2datetime ----------------------------')
    print(string2datetime('2017-03-15 08:10:10',"%Y-%m-%d %H:%M:%S.%f"))
    '''