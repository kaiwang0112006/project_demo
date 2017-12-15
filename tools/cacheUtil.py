# -*-coding: utf-8-*-

import functools
import os
import pickle

def pickle_cache(file):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            #print('%s %s():' % (text, func.__name__))
            if os.path.exists(file):
                with open(file,'rb') as fin:
                    result = pickle.load(fin)
            else:
                result = func(*args, **kw)
                with open(file,'wb') as fin:
                    pickle.dump(result,fin,True)
            return result
        return wrapper
    return decorator

if __name__ == '__main__':
    @pickle_cache(file=r'D:\\test.pkl')
    def test():
        return [1,2,4]
    
    print(test())