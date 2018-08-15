import numpy as np 
from math import sqrt
import matplotlib.pyplot as plt 
import warnings 
from matplotlib import style 
from collections import Counter
import pandas as pd 
from numpy import random

style.use('fivethirtyeight')


def k_n_n(data, predict, k=3):
	if len(data) >= k :
		warnings.warn("k value less than total voting")
	distances = [] # list of distances 

	for group in data:
		for features in data[group]:
			ec_distance = np.linalg.norm(np.array(features)- np.array(predict)) 
			distances.append([ec_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]] # votes for the distances sorted in order

	vote_result = Counter(votes).most_common(1)[0][0] # counter is the dict subclass to count the hashable objects, .most_common() is used to count the most common votes in the list. [0] means the top one.

	return vote_result

df = pd.read_csv('BCancer.data')

# we have some missing data replaced by '?' in the dataset
df.replace('?', -99999, inplace=True) # -99999 is to keep the data as a large outlier

#no use of id column in the prediction of the tumor
df.drop(['id'], 1, inplace=True)

full_data = df.astype(float).values.tolist() # clean data into float

random.shuffle(full_data)
test_size = 0.2 
train_set =  {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)): ]

for i in train_data:
	train_set[i[-1]].append(i[:-1])

for i in test_data:
	test_set[i[-1]].append(i[:-1])

correct =0
total = 0

for group in test_set:
	for data in test_set[group]:
		vote = k_n_n(train_set, data, k=5)
		if group == vote:
			correct += 1
		total +=1
print ( correct/total)
