import matplotlib.pyplot as plt
import numpy as np
import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import os
import math
import re
import nltk
from nltk.corpus import stopwords
import string
import csv

#data preprocess
positive_path = './train/pos'
negative_path = './train/neg'

positive_data_list = [] #text and result pair
negative_data_list = []
positive_freq_map = {}
negative_freq_map = {}

positive_theta_map = {}
negative_theta_map = {}

def process_files(path):
    for filename in os.listdir(path):
        # print(path+'/'+filename)
        process_file(path,filename)

def process_file(path,filename):
    global positive_data_list
    global negative_data_list
    with open(path + '/' + filename) as fp:
        if(path is positive_path):
            positive_data_list.append(fp.read())
        else:
            negative_data_list.append(fp.read())
    fp.close()

def build_freq_map(list,pos):
    global positive_freq_map
    global negative_freq_map
    global positive_theta_map
    global negative_theta_map
    for sentence in list:
        words = re.split('; |, |\*|\n| |,|;|\. |\.|\(|\)|\{|\}',sentence)
        for word in words:
            if pos is 1:
                if word in positive_freq_map:
                    positive_freq_map[word] = positive_freq_map[word] + 1
                else:
                    positive_freq_map[word] = 1
            else:
                if word in negative_freq_map:
                    negative_freq_map[word] = negative_freq_map[word] + 1
                else:
                    negative_freq_map[word] = 1

        for word in set(words):
            if pos is 1:
                if word in positive_theta_map:
                    positive_theta_map[word] = positive_theta_map[word] + 1
                else:
                    positive_theta_map[word] = 1
            else:
                try:
                    negative_theta_map[word] += 1
                except:
                    negative_theta_map[word] = 1

def clean_doc(doc):
    tokens = doc.lower().split()
    table = str.maketrans('','',string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word)>1]
    return tokens

def merge_freq_map():
    result = {key: positive_freq_map.get(key, 0) + negative_freq_map.get(key, 0)
          for key in set(positive_freq_map) | set(negative_freq_map)}
    return result

def get_difference():
    set1 = set(positive_freq_map.keys())
    set2 = set(negative_freq_map.keys())
    return set1 - set2

def get_most_freq_words(freq_map, low, high):
    frequency_map_sorted = sorted(freq_map.items(), key=lambda kv: kv[1], reverse = True)
    return [i[0] for i in frequency_map_sorted[low:high]]

process_files(positive_path)
process_files(negative_path)

build_freq_map(positive_data_list,1)
build_freq_map(negative_data_list,0)

freq_map = merge_freq_map()

freq_words = get_most_freq_words(freq_map,0,1000)

features = freq_words

#algorithm

def predict(theta_1, f,features):
    r1 = math.log(theta_1/(1-theta_1))
    r2 = 0
    for index,feature in enumerate(f):
        # print(features[index])
        theta_j1 = calculate_theta_j1(features[index])
        # print(theta_j1)
        theta_j0 = calculate_theta_j0(features[index])
        tmp = feature * math.log(theta_j1/theta_j0) + (1-feature)*math.log((1-theta_j1)/(1-theta_j0))
        r2 = r2 + tmp
        # print("r2: ",r2)
    return r1 + r2

# def predict(theta_1, f, features):
#      logdiv = math.log(theta_1/(1-theta_1))
#      xTw=0
#      Sumwj_0=0
#
#      for j in range(len(f)):
#
#         theta_j1=calculate_theta_j1(features[j])
#         theta_j0=calculate_theta_j0(features[j])
#
#         #print("actual theta_i1: ",theta_j1)
#         #print("actual thata_j0: ",theta_j0)
#         wj_0=math.log((1-theta_j1)/(1-theta_j0))
#         wj_1=math.log((theta_j1)/(theta_j0))
#         xTw=xTw+(wj_1-wj_0)*(f[j])#f[j] is x[j]
#         #print("xTw: ",xTw)
#         Sumwj_0=wj_0+Sumwj_0
#
#
#      W0=logdiv+Sumwj_0
#      y=W0+xTw
#      return y

def calculate_theta_1():
    return len(positive_data_list)/(len(positive_data_list)+len(negative_data_list))

def calculate_theta_j1(feature):
    if feature in positive_freq_map:
        # if positive_freq_map[feature]/len(positive_data_list) > 1:
        #    if ((positive_freq_map[feature]/len(positive_data_list))-1)*0.01+0.9501<1:
        #         return ((positive_freq_map[feature]/len(positive_data_list))-1)*0.01+0.95001
        #    else:
        #         return 0.9901
        # else:
        # print("j1 ", feature, " : ", (positive_theta_map[feature]+1)/(len(positive_data_list)+2))
        return (positive_theta_map[feature]+1)/(len(positive_data_list)+2)
    else:
        return 1/(len(positive_data_list)+2)


def calculate_theta_j0(feature):
    if feature in negative_freq_map:
        # if (negative_freq_map[feature]/len(negative_data_list) > 1):
        #    if ((negative_freq_map[feature]/len(negative_data_list))-1)*0.01+0.95<1:
        #         return ((negative_freq_map[feature]/len(negative_data_list))-1)*0.01+0.95
        #    else:
        #         return 0.9902
        # else:
        # print("j0 ", feature, " : ", (negative_theta_map[feature]+1)/(len(negative_data_list)+2))
        return (negative_theta_map[feature]+1)/(len(negative_data_list)+2)
    else:
        return 1/(len(negative_data_list)+2)


#
# #predict
with open('some.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows([['Id','Category']])
    i = 0
    while True:
    # for filename in os.listdir('./train/pos'):
        text = ""
        with open('./test' + '/' + str(i) + '.txt') as fp:
        # with open('./train/pos' + '/' + filename) as fp:
            text = fp.read()
        fp.close()

        words = re.split('; |, |\*|\n| |,|;|\. |\.|\(|\)|\{|\}',text)
        # print(words)

        f = [0]*(1000)
        for idx,feature in enumerate(features):
            if feature in words:
                f[idx] = 1

        theta_1= calculate_theta_1()
        # print(theta_1)

        y_combined = predict(theta_1,f, features)

        print("y_combined", y_combined)
        if(y_combined > 0):
            writer.writerows([[i,1]])
            # print("file " + str(i) + ": 1")
        else:
            writer.writerows([[i,0]])
            # print("file " + str(i) + ": 0")
        i+=1

        if(i>24999):
            break

#
