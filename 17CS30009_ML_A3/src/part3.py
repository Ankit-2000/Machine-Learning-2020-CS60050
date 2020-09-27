# Ankit Bagde - 17cs30009
# Task 3 - KMeans Clustering

import math
import pandas as pd 
import numpy as np

from part2 import write_txt

def kmeans(data):

	# randomly initializing 8 cluster centroids
	n_clusters = 8
	points = np.random.randint(590,size=8)
	cluster_centroids = np.array(data[points])
	prev_err = 1*math.inf

	# example***************to test
	# uncomment below code and euclidian distance line
	# http://mnemstudio.org/clustering-k-means-example-1.htm
	
	# data = np.zeros((7,2))
	# data = np.array([[1.0, 1.0], [1.5, 2.0], [3.0, 4.0], [5.0, 7.0], [3.5, 5.0], [4.5, 5.0], [3.5, 4.5]])
	# cluster_centroids = np.array([[1.0, 1.0], [5.0, 7.0]])
	# n_clusters = 2
	
	#repeat	
	while(1):

		# cluster assignment
		clusters = [[] for x in range(n_clusters)]
		err = 0
		for i in range(data.shape[0]):
			min_ = 1*math.inf
			for j in range(n_clusters):
				temp = math.exp(-np.dot(cluster_centroids[j],data[i]))
				# temp = np.linalg.norm(cluster_centroids[j] - data[i])  # for testing purpose
				if(min_ > temp):
					min_ = temp
					index = j
			clusters[index].extend([i])
			err = err + math.exp(-np.dot(cluster_centroids[index],data[i]))

		# error 
		if(abs(prev_err - err) < 0.000001 and prev_err >= err):
			break
		prev_err = err
		
		# move centroid
		for i in range(n_clusters):
			sum_ = np.zeros((1,data.shape[1]))
			for j in range(len(clusters[i])):
				sum_ = sum_ + data[clusters[i][j]]
			cluster_centroids[i] = sum_/len(clusters[i])

	return clusters

# main function
def main():
	data = pd.read_csv("../data/tf_idf_part1.csv")
	labels = data['labels']
	data.drop(['labels'],1, inplace=True)
	data = data.values
	
	clusters = kmeans(data)

	# write to "kmeans.txt"
	write_txt(clusters, "kmeans.txt")

if __name__ == "__main__":
	main()
