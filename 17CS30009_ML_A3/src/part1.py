# Ankit Bagde - 17cs30009
# Task 1 - Dataset Preparation

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

data = pd.read_csv("../data/AllBooks_baseline_DTM_Labelled.csv")
print("Data shape is %s"%(str(data.shape)))

# replace the labels with only names of the religious text
data['Unnamed: 0'] = [re.sub(r"_.*","",x) for x in data['Unnamed: 0']]

# drop the 13th index row
data.drop([13],0, inplace=True)
data = data.reset_index().drop(['index'],1)

print("Modified Data shape is %s"%(str(data.shape)))

# manual implemantation of tf-idf matrix construction
def tf_idf(a):
	n = a.shape[0]

	df = np.zeros((a.shape[1],))
	for t in range(a.shape[1]):
		for d in range(a.shape[0]):
			if(a[d][t]>0):
				df[t] = df[t] + 1

	tf = a * ((np.log((1+n)/(1+df))))

	for i in range(tf.shape[0]):
		tf[i] = tf[i]/np.linalg.norm(tf[i])

	return tf

labels = data['Unnamed: 0']
data.drop(['Unnamed: 0'],1, inplace=True)
terms = list(data)
data = data.values

##### manual implemantation
x = tf_idf(data)

# using scikit learn
# transformer = TfidfTransformer(norm='l2', smooth_idf=True)
# x = transformer.fit_transform(data)
# x = np.array(x.todense())

# save tf_idf matrix as df
df = pd.DataFrame()
df['labels'] = labels
for i in range(len(terms)):
	df[terms[i]] = x[:,i]

# df to csv
df.to_csv('../data/tf_idf_part1.csv', index=False)
print("TF-IDF matrix saved as dataframe to 'tf_idf_part1.csv' in 'data' folder\n\n")