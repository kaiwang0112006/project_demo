# -*-coding: utf-8-*-
import hashlib
import traceback

def maskstr(s):
    try:
        m = hashlib.md5(s.encode('utf8'))
    except:
        #traceback.print_exc()
        print(s)
    return m.hexdigest()

def idstr(s):
    try:
        return(str(int(float(s))))
    except:
        #traceback.print_exc()
        return str(s)

def masktry(s):
    s = s.upper()
    s =  idstr(s)
    if s.lower()!= 'nan':
        return maskstr(s)
    return s

if __name__ == '__main__':
    print(masktry('13609591632.0'))