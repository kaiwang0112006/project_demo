# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pymongo
import traceback
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
