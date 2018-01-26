# -*-coding: utf-8-*-
import pandas as pd
from sas7bdat import SAS7BDAT

class SAS7BDAT_free(SAS7BDAT):
    def head_to_data_frame(self,n):
        count = 0
        data = []
        for line in self.readlines():
            data.append(line)
            count += 1
            if count > n:
                break
        return pd.DataFrame(data[1:], columns=data[0])
    
    def linebyline2file(self,filename):
        with open(filename,'w') as fout:
            for eachline in self.readlines():
                line = [str(i) for i in eachline]
                fout.write(",".join(line))
