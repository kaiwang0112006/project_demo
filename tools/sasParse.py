# -*-coding: utf-8-*-
class SAS7BDAT_self(SAS7BDAT):
    def head_to_data_frame(self,n):
        import pandas as pd
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
            for line in self.readlines():
                fout.write(",".join(line))