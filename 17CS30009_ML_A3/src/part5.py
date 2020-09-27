# Ankit Bagde - 17cs30009
# Task 5 - Evaluation of the clusters

import os
import numpy as np
import pandas as pd

# Referred from - https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf

def entropy_class_labels(m,labels):
	counts = labels.value_counts()
	C = {x:k for (x,k) in zip(counts.index,counts)}
	P = {x:k/m for (x,k) in zip(counts.index,counts)}
	H = 0
	for i in P.values():
		H = H - i*np.log(i)
	return H

def entropy_cluster_labels(clusters, m, n_clusters):
	H = 0
	for i in range(n_clusters):
		x = len(clusters[i])/m
		H = H - x*np.log(x)
	return H

def conditional_entropy(clusters, m, labels, n_clusters):
	H = 0
	for i in range(n_clusters):
		p = len(clusters[i])/m
		
		# a dict of labels to hold no. of labels of each type in a cluster
		dic = {x:0 for x in labels}
		for j in clusters[i]:
			dic[labels[int(j)]] = dic[labels[int(j)]] + 1
		
		h_yc = 0
		for j in dic.values():
			if(j/len(clusters[i])!=0):
				h_yc = h_yc + (j/len(clusters[i]))*np.log(j/len(clusters[i]))

		h_yc = (h_yc * -p)
		H = H + h_yc
	return H


def test():

	print("-----------	TEST case	-----------")
	labels = [1,1,1,2,2,2,3,3,3,3,1,1,2,2,2,2,2,2,2,3]
	clusters = [[0,1,2,3,4,5,6,7,8,9], [10,11,12,13,14,15,16,17,18,19]]


	H_class = entropy_class_labels(20, pd.Series(labels)) / np.log(2)
	print("Entropy of class labels H(Y)  -   %f"%(H_class))
	
	H_cluster = entropy_cluster_labels(clusters, 20, 2) / np.log(2)
	print("Entropy of class labels H(C)  -   %f"%(H_cluster))

	# Mutual Info I = H_class - (entropy of class labels within each cluster)
	I = H_class - (conditional_entropy(clusters, 20, labels, 2) / np.log(2))
	print("Mutual Information I(Y;C)  	 -   %f"%(I))

	NMI = 2*I / (H_class + H_cluster)
	print("NMI score is %f"%(NMI))
	print("--------------------------------------------\n\n")



def main():

	n_clusters = 8

	# # # example***************to test
	# # https://course.ccs.neu.edu/cs6140sp15/7_locality_cluster/Assignment-6/NMI.pdf
	# # uncomment below function call
	# test()

	data = pd.read_csv("../data/tf_idf_part1.csv")
	labels = data['labels']
	data.drop(['labels'],1, inplace=True)
	data = data.values

	H_class = entropy_class_labels(data.shape[0],labels)

	for file in os.listdir("../clusters"):
		
		f = open("../clusters/"+file,'r')
		clusters_ = f.read().split('\n')
		clusters = [clusters_[i].split(', ') for i in range(8)]

		H_cluster = entropy_cluster_labels(clusters, data.shape[0], n_clusters)

		# Mutual Info I = H_class - (entropy of class labels within each cluster)
		I = H_class - conditional_entropy(clusters, data.shape[0], labels, n_clusters)

		NMI = 2*I / (H_class + H_cluster)
		print("NMI score for '%s' is %.13f"%(file,NMI))


if __name__ == "__main__":
	main()