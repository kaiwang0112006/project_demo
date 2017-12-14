# -*-coding: utf-8-*-
import jieba
import pandas as pd
import copy
from .multi_apply import *
import os
import jieba.posseg as pseg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TAOBAOFILE = os.path.join(BASE_DIR, 'userdicts', 'taobao.txt')
STOPWORDSFILE = os.path.join(BASE_DIR, 'userdicts', 'stopwords.txt')

stopwords = {}
# with open(STOPWORDSFILE) as fin:
#     for eachline in fin:
#         stopwords[eachline.strip()] = 1
# 
# jieba.load_userdict(TAOBAOFILE)

def jieba_cut(sentence,keywords=[],empty='NA'):
    '''
    cut sentence with jieba and only keep keywords filtered by word count.
    '''
    
    num_list = ['0','1','2','3','4','5','6','7','8','9']
    try:    
        st = [s for s in sentence.decode('utf-8') if s not in num_list and (u'\u4e00' <= s <= u'\u9fff')]
    except:
        return ['NA']
    s_in = ''.join([s for s in st])
    seg = [s for s in list(jieba.cut(s_in.encode('utf-8'))) if len(s)>1]
    if len(keywords)>0:
        seg_list = [s for s in seg if s in keywords] # 只保留关键词
    else:
        seg_list = copy.deepcopy(seg)

    if len(seg_list) == 0:
        seg_list = [empty]
    
    return seg_list

def jieba_str(x,keywordsdict={}):
    '''
    format jieba cut output to string.
    '''
    seg = jieba_cut(x,keywordsdict)
    segr = [i.encode('utf-8') for i in seg]
    return ','.join(segr)

def jieba_cut_3(sentence,keywords=[],empty='NA',stopwordsfile='',definefile=''):
    '''
    cut sentence with jieba and only keep keywords filtered by word count.
    '''
    stopwords = {}
    if stopwordsfile:
        with open(STOPWORDSFILE) as fin:
            for eachline in fin:
                stopwords[eachline.strip()] = 1
                
    if definefile:
        jieba.load_userdict(TAOBAOFILE)
        
    
    num_list = ['0','1','2','3','4','5','6','7','8','9']
    seg = [s for s in list(jieba.cut(sentence)) if len(s)>1 and s not in stopwords and (u'\u4e00' <= s <= u'\u9fff') and s not in num_list]

    if len(keywords)>0:
        seg_list = [s for s in seg if s in keywords] # 只保留关键词
    else:
        seg_list = copy.deepcopy(seg)
    
    if len(seg_list) == 0:
        if empty == '':
            seg_list = []
        else:
            seg_list = [empty]

    return seg_list

def noun_filter(words):
    segs = []
    for word in words:
        wcuts = pseg.cut(word)
        for w,f in wcuts:
            if f[0] == 'n' and len(w)>1:
                segs.append(w)
    return segs

def jieba_cut_3_noun(sentence,keywords=[],empty='NA'):
    '''
    cut sentence with jieba and only keep keywords filtered by word count.
    '''
    
    num_list = ['0','1','2','3','4','5','6','7','8','9']

    seg = [s for s in list(jieba.cut(sentence)) if len(s)>1 and s not in stopwords and (u'\u4e00' <= s <= u'\u9fff') and s not in num_list]
    if len(keywords)>0:
        seg_list = [s for s in seg if s in keywords] # 只保留关键词
    else:
        seg_list = copy.deepcopy(seg)

    seg_list = noun_filter(seg_list)
    if len(seg_list) == 0:
        if empty == '':
            seg_list = []
        else:
            seg_list = [empty]

    return seg_list


def jieba_str_3(x,keywordsdict={}):
    '''
    format jieba cut output to string
    '''
    print(x)
    seg = jieba_cut_3(x,keywordsdict)
    segr = [i for i in seg]
    return ','.join(segr)

def filter_keywords_by_stopwords(keywords):
    '''
    defined keyword that filtered words by idf & count.
    '''

    keywordsdict = {}
    for w in keywords:
        if w not in stopwords:
            try:
                keywordsdict[w] = 1
            except:
                pass
    return keywordsdict

def filter_keywords_by_stopwordsi_p2(keywords):
    '''
     defined keyword that filtered words by idf & count. (written for python2)
    '''
            
    keywordsdict = {}
    for w in keywords:
        if w not in stopwords:
            try:
                keywordsdict[w.decode('utf-8')] = 1
            except:
                pass
    return keywordsdict

def xjieba_str(row, key):
    return jieba_str(row.get(key, ''))

def xjieba_cut(row, key):
    return jieba_cut(row.get(key, ''))

def xjieba_cut_3(row, key):
    return jieba_cut_3(row.get(key, ''))

def fenci_pd(iinput,output,sep_in,sep_out,nrows=None):
    tb = pd.read_csv(iinput,sep=sep_in,nrows=nrows)
    tb['words_str'] = tb['product_name'].apply(lambda x:jieba_str(x))
    tb.to_csv(output,sep=sep_out,index=False)

def fenci_pd_p3_sas(iinput,output,sep_in,sep_out,nrows=None):
    tb = pd.read_csv(iinput,sep=sep_in,nrows=nrows)
    tb['words'] = tb['goods_name'].apply(lambda x:jieba_cut_3(x))
    tb.to_csv(output,sep=sep_out,index=False)
    return tb

def fenci_pd_p3_sas_multi(iinput,output,sep_in,sep_out,nrows=None):
    tb = pd.read_csv(iinput,sep=sep_in,nrows=nrows)
    tb['words'] = apply_by_multiprocessing(tb, xjieba_cut_3, key='goods_name', axis=1)
    #b.to_csv(output,sep=sep_out,index=False)
    return tb


def fenci_pd_multi(iinput,output,sep_in,sep_out,nrows=None):
    tb = pd.read_csv(iinput,sep=sep_in,nrows=nrows)
    tb['words_str'] = apply_by_multiprocessing(tb, xjieba_str, key='product_name', axis=1)
    tb.to_csv(output,sep=sep_out,index=False)
    
