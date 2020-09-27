# Ankit Bagde - 17cs30009
# Task 1 - Dataset Generation

import pandas as pd 
import numpy as np

# read csv
data_a = pd.read_csv("../data/winequality-red.csv", sep=';')
data_b = pd.read_csv("../data/winequality-red.csv", sep=';')
print("data size is : "+str(data_a.shape))

######## dataset A

# quality attribute
data_a['quality'] = [0 if x<=6 else 1 for x in data_a['quality']]

# min-max scaling
for i in data_a:
	if(i!='quality'):
		min_ele = np.min(data_a[i])
		max_ele = np.max(data_a[i])
		data_a[i] = (data_a[i]-min_ele)/(max_ele - min_ele)

# save csv as dataset_A.csv
data_a.to_csv('../data/dataset_A.csv', index=False)
print("Dataset A generated. File saved as dataset_A.csv in 'data' folder")

######## dataset B

# quality attribute
data_b['quality'] = [0 if x<5 else 1 if x==5 or x==6 else 2 for x in data_b[i]]

# Z-score normalization
for i in data_b:
	if(i!='quality'):
		mean = np.mean(data_b[i])
		std = np.std(data_b[i])
		data_b[i] = (data_b[i]-mean)/std

# bin segregation
for i in data_b:
	if(i!='quality'):
		min_ele = np.min(data_b[i])
		max_ele = np.max(data_b[i])
		diff = (max_ele - min_ele)/4
		data_b[i] = [0 if x<min_ele+diff else 1 if x<min_ele+2*diff else 2 if x<min_ele+3*diff else 3 for x in data_b[i]]

# save csv as dataset_B.csv
data_b.to_csv('../data/dataset_B.csv', index=False)
print("Dataset B generated. File saved as dataset_B.csv in 'data' folder")
