import numpy as np
import xml
import sys
import xml.etree.ElementTree as ET
import math
import pickle
import re
import jieba
def remove_punctuation(line):
    rule = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
    line = rule.sub('',line)
    return line

class TF_IDF():
    def __init__(self, query, file_list, invert_file, voc_file, doc_length, stopwords=None, narr=False):
        self.docs = {}
        self.tf = {}
        self.df = {}
        self.idf = {}
        self.word_voc = []
        self.query = query
        self.invert = invert_file
        self.file_name = file_list
        self.ori_voc = voc_file
        self.doc_length = doc_length
        self.narrative = narr
        self.stopwords = stopwords
        self.cal_word_id()
        #self.word_id = word_id
        #self.cal_tfidf()
    
    def cal_word_id(self):
        count = 1
        self.word_id = {}
        self.id_word = {}
        for word in self.ori_voc:
            if word == 'utf8\n':
                print('Skip')
                continue
            self.word_id[word[:-1]] = count
            self.id_word[count] = word[:-1]
            count += 1
            
    def cal_tfidf(self):
        num = 0
        if self.narrative:
            for elem in self.query.iter('narrative'):
                narrative = elem.text[1:-2].split('。')
                #sen = elem.text[3:-1]
                #print(sen)
                if len(narrative) > 1:
                    narrative = narrative[:-1]
                for sen in narrative:
                    sen = sen.replace('相關文件內容', '')
                    sen = sen.replace('包括', '') 
                    sen = sen.replace('應', '')
                    sen = sen.replace('說明', '')
                    sen = remove_punctuation(sen)
                    sen = jieba.cut(sen)
                    words = " ".join(sen).split()
                    #words = list(filter(lambda a: a not in self.stopwords and a != '\n', words))
                    for word in words:
                        if len(word) == 1:
                            continue
                        for i in range(len(word)-1):
                            if word[i:i+2] not in self.word_voc:
                                self.word_voc.append(word[i:i+2])
                    #for i in range(len(sen)-1):
                    #    if sen[i:i+2] not in self.word_voc:
                    #        self.word_voc.append(sen[i:i+2]) 
        
        for elem in self.query.iter('concepts'):
            query_list = elem.text[1:-2].split('、')
            #query_list = list(filter(lambda a: a not in self.stopwords and a != '\n', query_list))
            for word in query_list:
                if len(word) == 2:
                    if word not in self.word_voc:
                        self.word_voc.append(word)
                else:
                    for i in range(len(word)-1):
                        if word[i:i+2] not in self.word_voc:
                            self.word_voc.append(word[i:i+2])
        '''
        for elem in self.query_test.iter('concepts'):
            query_list = elem.text[1:-2].split('、')
            for word in query_list:
                if len(word) == 2:
                    if word not in self.word_voc:
                        self.word_voc.append(word)
                else:
                    for i in range(len(word)-1):
                        if word[i:i+2] not in self.word_voc:
                            self.word_voc.append(word[i:i+2])
        '''
        #print(self.voc)
     
        while num < len(self.invert):
            #print(self.invert[num])
            term, term2, app_doc = self.invert[num].split()
            term, term2, app_doc = int(term), int(term2), int(app_doc)
            word = self.id_word[term]
            #print(word)
            if term2 == -1:
                num += (app_doc+1)
                continue
            else: 
                word2 = self.id_word[term2]
                #print(word+word2)
                if word + word2 not in self.word_voc:
                    num += (app_doc+1)
                    continue
                if word + word2 not in self.df:
                    self.df[word+word2] = app_doc
                else:
                    self.df[word+word2] += app_doc
                
                for j in range(app_doc):
                    num += 1
                    doc_num, times = self.invert[num].split()
                    times = int(times)
                    doc_num = int(doc_num)
                    if word+word2 not in self.tf:
                        tmp = {}
                        tmp[doc_num] = times
                        self.tf[word+word2] = tmp
                    else:
                        if doc_num not in self.tf[word+word2]:
                            self.tf[word+word2][doc_num] = times
                        else:
                            self.tf[word+word2][doc_num] += times
                    
            num += 1 

        for word, df in self.df.items():
            self.idf[word] = math.log10(46972 / df)
    
    def tf_idf(self, index, word):
        k1 = 1.8
        b = 0.75
        if index in self.tf[word]:
            return self.tf[word][index]*self.idf[word]* (k1+1) / (self.tf[word][index] + k1 * (1-b+b*self.doc_length[index]/self.doc_length['avg']))
        else:
            return 0
    
    def get_doc_vector(self, index):
        return  [1*self.tf_idf(index, w) if w in self.tf else 0 for w in self.word_voc]
    

