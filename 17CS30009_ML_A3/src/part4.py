# Ankit Bagde - 17cs30009
# Task 4 - Attribute Reduction by Principal Component Analysis

import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA

from part2 import agglomerative, write_txt
from part3 import kmeans

data = pd.read_csv("../data/tf_idf_part1.csv")
labels = data['labels']
data.drop(['labels'],1, inplace=True)
data = data.values

# apply PCA
pca = PCA(n_components=100)
data = pca.fit_transform(data)

for i in range(data.shape[0]):
	data[i] = data[i]/np.linalg.norm(data[i])

# agglomerative clustering 
clusters = agglomerative(data)
write_txt(clusters, "agglomerative_reduced.txt")

# kmeans clustering
clusters = kmeans(data)
write_txt(clusters, "kmeans_reduced.txt")

