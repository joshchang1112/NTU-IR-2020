import numpy as np
import xml
import sys
import xml.etree.ElementTree as ET
import math
import pickle
import re
from tfidf import TF_IDF, remove_punctuation
import jieba
from argparse import ArgumentParser

'''
stopWords=[]
with open('stopWords.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stopWords.append(data)
'''

parser = ArgumentParser(description='vsmodel')
parser.add_argument('-r', '--relevance_feedback', action='store_true', help="using Rocchio")
parser.add_argument('-i', type=str, default="queries/query-train.xml", dest="query_file")
parser.add_argument('-o', type=str, default="myresults.csv", dest="ranked_list")
parser.add_argument('-m', type=str, default="model", dest="model_dir")
parser.add_argument('-d', type=str, dest="NTCIR_dir")
args = parser.parse_args()

query = args.query_file
output = args.ranked_list
model = args.model_dir
file_name = model+'/file-list'
with open(file_name, 'r') as f:
    file_list = f.readlines()

# Calculate doc length
avg_doc_length = 0
doc_length = {}
for i in range(len(file_list)):
    with open(args.NTCIR_dir+'/'+file_list[i][:-1], 'r', encoding='utf-8') as xml_file:
        count = 0
        article = ET.parse(xml_file)
        for elem in article.iter('p'):
            count += len(elem.text)
    doc_length[i] = count
    avg_doc_length += count

avg_doc_length /= len(file_list)
doc_length['avg'] = avg_doc_length
#print(doc_length['avg'])
print("Finish cal doc length")

# Load query, model file
with open(query, 'r', encoding='utf-8') as xml_file:
    query = ET.parse(xml_file)

invert = model+'/inverted-file'
with open(invert, 'r') as f:
    invert_file = f.readlines()
voc = model+'/vocab.all'   
with open(voc, 'r') as f:
    voc_file = f.readlines()

tfidf = TF_IDF(query, file_list, invert_file, voc_file, doc_length, narr=True)
print("Start Calculate TFIDF")
tfidf.cal_tfidf()
print("Voc size:{}".format(len(tfidf.word_voc)))
doc_vector = []
for i in range(len(file_list)):
    doc_vector.append(tfidf.get_doc_vector(i))
doc_vector = np.array(doc_vector)
#np.save('doc_vector.npy', doc_vector)
print("Finish Calculating Document vector successfully:)")
'''
tfidf2 = TF_IDF(query, file_list, invert_file, voc_file, doc_length)
print("Start Calculate TFIDF")
tfidf2.cal_tfidf()
print("Voc size:{}".format(len(tfidf2.word_voc)))
doc_vector2 = []
for i in range(len(file_list)):
    doc_vector2.append(tfidf2.get_doc_vector(i))
doc_vector2 = np.array(doc_vector2)
#np.save('doc_vector.npy', doc_vector)
print("Finish Calculating Document vector successfully:)")
'''
# Part II: Query
query_tf = {}
concept_tf = {}
count = 0
root = query.getroot()
for elem in root.iter('narrative'):
    narrative = elem.text[1:-2].split('。')
    #print(narrative)
    #sen = elem.text[3:-1]
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
        #words = list(filter(lambda a: a not in stopWords and a != '\n', words))
        words_idf = {}
        for word in words:
            if len(word) == 1:
                continue
            for i in range(len(word)-1):
                if word[i:i+2] not in words_idf and word[i:i+2] in tfidf.idf:
                    words_idf[word[i:i+2]] = tfidf.idf[word[i:i+2]]
        for word in words:
            for i in range(len(word)-1):
                if word[i:i+2] in words_idf:
                    if words_idf[word[i:i+2]] > 0:
                        if word[i:i+2] not in query_tf: 
                            query_tf[word[i:i+2]] = {count: 1}
                        else:
                            query_tf[word[i:i+2]][count] = 1


    count += 1

count = 0
for elem in root.iter('concepts'):
    query_list = elem.text[1:-2].split('、')
    #query_list = list(filter(lambda a: a not in stopWords and a != '\n', query_list))
    for word in query_list:
        for i in range(len(word)-1):
            if word[i:i+2] not in query_tf:
                query_tf[word[i:i+2]] = {count: 1}
            else:
                if count not in query_tf[word[i:i+2]]:
                    query_tf[word[i:i+2]][count] = 1
                else:
                    query_tf[word[i:i+2]][count] += 1
            
            if word[i:i+2] not in concept_tf:
                concept_tf[word[i:i+2]] = {count: 1}
            else:
                if count not in concept_tf[word[i:i+2]]:
                    concept_tf[word[i:i+2]][count] = 1
                else:
                    concept_tf[word[i:i+2]][count] += 1
    count += 1

doc_count = count
rel_num = 10
index_list = []
index_list2 = []
index_list_sum = []
print(doc_vector.shape)
print("Calculate cosine similarity")
for i in range(doc_count):
    query_vec = []
    for w in tfidf.word_voc:
        if w in tfidf.idf and w in query_tf:
        #if w in tfidf.idf and w in concept_tf:
            #if i in concept_tf[w]:
            if i in query_tf[w]:
                query_vec.append(query_tf[w][i] * tfidf.idf[w])
                #query_vec.append(concept_tf[w][i] * tfidf.idf[w])
            else:
                query_vec.append(0)
        else:
            query_vec.append(0)
    query_vec = np.array(query_vec)
    
    cos_sim_list = []
    for j in range(len(doc_vector)):
        cos_sim = np.dot(doc_vector[j], query_vec) / (np.linalg.norm(doc_vector[j]) * np.linalg.norm(query_vec) + 1e-8)
        cos_sim_list.append(cos_sim)

    cos_sim_arr = np.array(cos_sim_list)
    ind = np.argpartition(cos_sim_arr, -100)[-100:]
    sort_ind = ind[np.argsort(cos_sim_arr[ind])][::-1]
    '''
    query_vec2 = []
    for w in tfidf2.word_voc:
        if w in tfidf2.idf and w in concept_tf:
            if i in concept_tf[w]:
                query_vec2.append(concept_tf[w][i] * tfidf2.idf[w])
            else:
                query_vec2.append(0)
        else:
            query_vec2.append(0)
    query_vec2 = np.array(query_vec2)

    cos_sim_list2 = []
    for j in range(len(doc_vector2)):
        cos_sim = np.dot(doc_vector2[j], query_vec2) / (np.linalg.norm(doc_vector2[j]) * np.linalg.norm(query_vec2) + 1e-8)
        cos_sim_list2.append(cos_sim)

    cos_sim_arr2 = np.array(cos_sim_list2)
    ind2 = np.argpartition(cos_sim_arr2, -100)[-100:]
    sort_ind2 = ind2[np.argsort(cos_sim_arr2[ind2])][::-1]
    #print(sort_ind)
    # Rocchio feedback
    '''
    if args.relevance_feedback:
        alpha, beta, gamma = 1, 0.35, 0.01
        rel_doc_vec = np.zeros((doc_vector.shape[1]))
        non_rel_doc_vec = np.zeros((doc_vector.shape[1]))
        count = 1e-8
        non_count = 0
        for j in range(len(doc_vector)):
            if j in sort_ind[:rel_num]:
            #if cos_sim_arr[j] > 0.7:
                rel_doc_vec += doc_vector[j] 
                count += 1
            #elif cos_sim_arr[j] < 0.2:
            else:
                non_rel_doc_vec += doc_vector[j]
                non_count += 1
        new_query_vec = alpha * query_vec + beta / count * rel_doc_vec - gamma / non_count * non_rel_doc_vec
        query_vec = new_query_vec
    
        # Second times
        cos_sim_list = []
        for j in range(len(doc_vector)):
            cos_sim = np.dot(doc_vector[j], query_vec) / (np.linalg.norm(doc_vector[j]) * np.linalg.norm(query_vec) + 1e-8)
            cos_sim_list.append(cos_sim)
    
        cos_sim_arr = np.array(cos_sim_list)
        ind = np.argpartition(cos_sim_arr, -100)[-100:]
        sort_ind = ind[np.argsort(cos_sim_arr[ind])][::-1]
    '''
    cos_sim_sum = 0.5*(cos_sim_arr) + 0.5*(cos_sim_arr2)
    ind_sum = np.argpartition(cos_sim_sum, -100)[-100:]
    sort_ind_sum = ind_sum[np.argsort(cos_sim_sum[ind_sum])][::-1]
    index_list_sum.append(sort_ind_sum)
    index_list2.append(sort_ind2)
    '''
    index_list.append(sort_ind)
    

'''
rank_list = []
for i in range(10):
    rank_appear = [0] * 46972
    rank_num = [1000] * 46972
    for j in range(len(index_list[i])):
        rank_appear[index_list[i][j]] += 1
        rank_appear[index_list2[i][j]] += 1
        rank_appear[index_list_sum[i][j]] += 1
        if rank_num[index_list[i][j]] != 1000:
            rank_num[index_list[i][j]] += j
        else:
            rank_num[index_list[i][j]] = j
        if rank_num[index_list2[i][j]] != 1000:
            rank_num[index_list2[i][j]] += j
        else:
            rank_num[index_list2[i][j]] = j
        if rank_num[index_list_sum[i][j]] != 1000:
            rank_num[index_list_sum[i][j]] += j
        else:
            rank_num[index_list_sum[i][j]] = j
    
    for j in range(len(file_list)):
        if rank_appear[j] == 1:
            rank_num[j] *= 5
        elif rank_appear[j] == 2:
            rank_num[j] *= 2
    
    rank = sorted(range(len(rank_num)), key=lambda k: rank_num[k])
    #rank_arr = np.array(rank_num[i])
    #sort_rank_ind = np.argpartition(rank_arr, 100)
    #sort_rank_ind = rank_ind[np.argsort(rank_arr[rank_ind])][::-1]
    rank_list.append(rank[:100])
'''
if doc_count == 10:
    _map = 0
    with open("queries/ans_train.csv", 'r') as f:
        ans = f.readlines()
        for i in range(1, 11):
            ap = 0
            count = 0
            target = ans[i].split(',')[1][:-1].split()
            for j in range(len(index_list[i-1])):
                if file_list[index_list[i-1][j]][:-1].split('/')[-1].lower() in target:
                    #print(j)
                    count += 1
                    ap += (count/(j+1))
            ap /= len(target)
            print(ap)
            _map += ap
    _map /= 10
    print("MAP:{}".format(_map))
else:
    with open("submit.csv", 'w') as f:
        f.write('query_id,retrieved_docs\n')
        #for i in range(len(index_list)):
        for i in range(10, len(index_list_sum)+10):
            if i < 9:
                f.write('00{},'.format(i+1))
            else:
                f.write('0{},'.format(i+1))
            for j in range(len(index_list_sum[i-10])):
                f.write(file_list[index_list_sum[i-10][j]][:-1].split('/')[-1].lower())
                if j != len(index_list[i-10])-1:
                    f.write(' ')
                else:
                    f.write('\n')
'''
_map = 0
with open("queries/ans_train.csv", 'r') as f:
    ans = f.readlines()
    for i in range(1, 11):
       ap = 0
       count = 0
       target = ans[i].split(',')[1][:-1].split()
       for j in range(len(index_list2[i-1])):
           if file_list[index_list2[i-1][j]][:-1].split('/')[-1].lower() in target:
               #print(j)
               count += 1
               ap += (count/(j+1))
       ap /= len(target)
       print(ap)
       _map += ap
_map /= 10
print("MAP:{}".format(_map))

_map = 0
with open("queries/ans_train.csv", 'r') as f:
    ans = f.readlines()
    for i in range(1, 11):
       ap = 0
       count = 0
       target = ans[i].split(',')[1][:-1].split()
       for j in range(len(index_list_sum[i-1])):
           if file_list[index_list_sum[i-1][j]][:-1].split('/')[-1].lower() in target:
               #print(j)
               count += 1
               ap += (count/(j+1))
       ap /= len(target)
       print(ap)
       _map += ap
_map /= 10
print("MAP:{}".format(_map))
'''
