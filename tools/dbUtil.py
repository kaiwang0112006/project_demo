# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pymongo
import traceback
import pyodbc
import pymysql

class MongoConnect(object):
    def __init__(self, host, port):
        self.host = host 
        self.port = port


    def connect(self):
        self.client = pymongo.MongoClient(self.host, self.port)
        
    def authdb(self, dbname, user, passwd):
        self.db = self.client[dbname]
        try:
            self.db.authenticate(user,passwd) 
        except:
            traceback.print_exc()
            print('auth failed')
        
    def close(self):
        self.client.close()

class mysqldbParse(object):
    def __init__(self, host='', port=3306, user='', passwd='', dbName='', charset='utf8',connect_timeout=31536000):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.dbName = dbName
        self.charset = charset
        self.connect_timeout = connect_timeout
        
    def conn_mysql(self):
        self.conn = pymysql.connect(host=self.host,
                                    port=self.port,
                                    user=self.user,
                                    password=self.passwd,
                                    db=self.dbName,
                                    charset=self.charset,
                                    connect_timeout = self.connect_timeout,
                                    cursorclass=pymysql.cursors.DictCursor)
        
    def insert_sql(self,insertSql):
        with self.conn.cursor() as cursor:
            cursor.execute(insertSql)
            self.conn.commit()

    def close(self):
        self.conn.close()
        
class sqlserverParse(object):
    def __init__(self, host='', user='', passwd='', dbName=''):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.dbName = dbName
        
    def conn_connect(self):
        self.conn = pyodbc.connect('DRIVER=(ODBC DRIVER 13 for SQL Server);SERVER='+self.host+';DATABASE='+
                                   self.dbName+';UID='+self.user+';PWD='+self.passwd)
        


    def close(self):
        self.conn.close()
