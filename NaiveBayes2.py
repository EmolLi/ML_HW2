import matplotlib.pyplot as plt
import numpy as np
import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import os
import math
import re

#data preprocess
positive_path = './train/pos'
negative_path = './train/neg'

positive_data_list = [] #text and result pair
negative_data_list = []
positive_freq_map = {}
negative_freq_map = {}

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


def merge_freq_map():
    result = {key: positive_freq_map.get(key, 0) + negative_freq_map.get(key, 0)
          for key in set(positive_freq_map) | set(negative_freq_map)}
    return result

def get_difference():
    set1 = set(positive_freq_map.keys())
    #print(set1)
    set2 = set(negative_freq_map.keys())
    return set1 - set2

def get_most_freq_words(freq_map, low, high):
    frequency_map_sorted = sorted(freq_map.items(), key=lambda kv: kv[1], reverse = True)
    return [i[0] for i in frequency_map_sorted[low:high]]

process_files(positive_path)
process_files(negative_path)
#print(positive_data_list)

build_freq_map(positive_data_list,1)
build_freq_map(negative_data_list,0)
#print(positive_freq_map)
#print(positive_freq_map.keys())


freq_map = merge_freq_map()
#print(freq_map)
freq_words = get_most_freq_words(freq_map,0,50)

features = freq_words

for i in range(len(features)):
    print(features[i] ,freq_map[features[i]])
#print(features)
print("---------------------------------------------------------------------")
print("\n")
rankpos_freq_words = get_most_freq_words(positive_freq_map,0,50)
#print(positive_freq_map)
print('what the fuck-------------------------------------')
for i in range(len(features)):
    print(rankpos_freq_words[i] ,positive_freq_map[features[i]])

print("---------------------------------------------------------------------")
rankneg_freq_words = get_most_freq_words(negative_freq_map,0,50)
for i in range(len(features)):
    print(rankneg_freq_words[i] ,negative_freq_map[features[i]])

#print(features[0])
#print("positive_freq_map[0]:",positive_freq_map[0])
#get_difference()
#print(negative_data_list)
features1 = rankpos_freq_words
features2 = rankneg_freq_words


def predict(theta_1, f, features):
     logdiv = math.log(theta_1/(1-theta_1))
     xTw=0
     Sumwj_0=0

     for j in range(len(f)):

        theta_j1=calculate_theta_j1(features[j])
        theta_j0=calculate_theta_j0(features[j])

        #print("actual theta_i1: ",theta_j1)
        #print("actual thata_j0: ",theta_j0)
        wj_0=math.log((1-theta_j1)/(1-theta_j0))
        wj_1=math.log((theta_j1)/(theta_j0))
        xTw=xTw+(wj_1-wj_0)*(f[j])#f[j] is x[j]
        #print("xTw: ",xTw)
        Sumwj_0=wj_0+Sumwj_0


     W0=logdiv+Sumwj_0
     y=W0+xTw
     return y


def calculate_theta_1():
     return len(positive_data_list)/(len(positive_data_list)+len(negative_data_list))

def calculate_theta_j1(feature):
     #print(positive_freq_map)
     for key in positive_freq_map:

        if feature==key:
          #print("found")
          #print("what?",feature)
          #print("key: ",key)
          #print(positive_freq_map[key])
          #print(positive_freq_map)
          #print(positive_freq_map[key])
          #print(len(positive_data_list))
          #print("lenpositive: ", len(positive_data_list))
          #print("theta_j1:", positive_freq_map[key]/len(positive_data_list))
          if (positive_freq_map[key]/len(positive_data_list) > 1):
            return 0.991
          else:
            return (positive_freq_map[feature]+1)/(len(positive_data_list)+2)
        #else:
        #   print("not found")

def calculate_theta_j0(feature):
    for key in negative_freq_map:
     if feature==key:
         #print("found")
         #print("what?",feature)
         #print("key: ",key)
         #print(negative_freq_map[key])

         #print("lennegative: ", len(negative_data_list))
         #print("theta_j0: ",negative_freq_map[feature]/len(negative_data_list))
         if (negative_freq_map[feature]/len(negative_data_list) > 1):
            return 0.992
         else:
            return (negative_freq_map[feature]+1)/(len(negative_data_list)+2)
     #else:
     #    print("not found")




 #print(most_freq_words)
 #print(len(positive_freq_map))
 #print(len(negative_freq_map))
 #print(len(freq_map))

#---------------------------------------------------------------------------
#predict


text = ""
with open('./train/neg' + '/' + '2_1.txt') as fp:
    text = fp.read()
fp.close()

words = re.split('; |, |\*|\n| |,|;|\. |\.|\(|\)|\{|\}',text)
# print(words)

f = [0]*(50)
for idx,feature in enumerate(features):
    if feature in words:
        f[idx] = 1

theta_1= calculate_theta_1()


ypos=predict(theta_1,f,features1)
yneg=predict(theta_1,f,features2)

print("ypos: ",ypos)
print("yneg: ",yneg)

if (ypos>yneg):
    print("predict: positive.")

else:
    print("predict: negative.")
# print(f)
# print(predict(calculate_theta_1(),f))
#X = []

# y = [1]*len(positive_data_list)
# y.append([0]*len(negative_data_list))
# print(X)





#
