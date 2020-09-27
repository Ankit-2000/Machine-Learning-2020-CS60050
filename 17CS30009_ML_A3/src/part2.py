# Ankit Bagde - 17cs30009
# Task 2 - Agglomerative Clustering

import math
import pandas as pd 
import numpy as np

def min_clusters(matrix,n_clusters):
	
	a = 1
	b = 0
	for i in range(n_clusters):
		for j in range(0,i):
			if(matrix[i][j]<matrix[a][b]):
				a = i
				b = j
	return a,b

# find minium distance of two clusters
def find_min(clus_a, clus_b, glb_matrix):
	min_ = 1*math.inf
	for i in clus_a:
		for j in clus_b:
			if(i>j and glb_matrix[i][j] < min_):
				min_ = glb_matrix[i][j]
			elif(i<j and glb_matrix[j][i] < min_):
				min_ = glb_matrix[j][i]
	return min_

def agglomerative(data):
	
	# each item considered as singleton cluster
	n_clusters = data.shape[0]
	clusters = []

	# initial cluster
	for i in range(n_clusters):
		clusters.append([i])

	# compute proximity matrix
	matrix = np.full((n_clusters,n_clusters), np.inf)

	for i in range(n_clusters):
		for j in range(0,i):
			matrix[i][j] = np.exp(-np.dot(data[i],data[j]))

	# example***************to test
	# uncomment and put n_clusters = 6 above
	# https://people.revoledu.com/kardi/tutorial/Clustering/Numerical%20Example.htm

	# matrix[1][0] = 0.71
	# matrix[2][0] = 5.66
	# matrix[2][1] = 4.95
	# matrix[3][0] = 3.61
	# matrix[3][1] = 2.92
	# matrix[3][2] = 2.24
	# matrix[4][0] = 4.24
	# matrix[4][1] = 3.54
	# matrix[4][2] = 1.41
	# matrix[4][3] = 1.00
	# matrix[5][0] = 3.20
	# matrix[5][1] = 2.50
	# matrix[5][2] = 2.50
	# matrix[5][3] = 0.50
	# matrix[5][4] = 1.12

	
	glb_matrix = matrix
	# repeat
	while(1):

		# find min distance clusters
		b,a = min_clusters(matrix,n_clusters)

		# merge closest two clusters
		clusters[a].extend(clusters[b])
		del clusters[b]

		# n_clusters decreased by 1 
		n_clusters = n_clusters - 1
		
		if(n_clusters <= 8):
			break		
		# delete row and column
		new_matrix = np.delete(matrix, b, 0)
		new_matrix = np.delete(new_matrix, a, 1)

		# modifying new_matrix
		for i in range(n_clusters):
			# changes in row 'a'
			if(i<a):
				new_matrix[a][i] = find_min(clusters[a],clusters[i], glb_matrix)
			# changes in column 'a'
			elif(i>a):
				new_matrix[i][a] = find_min(clusters[a],clusters[i], glb_matrix)

		# make new_matrix lower triangular matrix
		for i in range(n_clusters):
			for j in range(n_clusters):
				if(j>=i):
					new_matrix[i][j] = 1*math.inf

		matrix = new_matrix

	return clusters

def write_txt(clusters, name):
	f = open("../clusters/"+name,'w')
	for i in range(len(clusters)):
		clusters[i].sort()
		clusters = sorted(clusters, key=lambda x: x[0])
		f.write(str(clusters[i])[1:-1])
		f.write("\n")
	f.close()
	print("Clusters printed into '" + name + "' in 'clusters' folder\n\n")


def main():
	data = pd.read_csv("../data/tf_idf_part1.csv")
	labels = data['labels']
	data.drop(['labels'],1, inplace=True)
	data = data.values

	print("Data shape is %s"%(str(data.shape)))

	clusters = agglomerative(data)

	# write to "agglomerative.txt"
	write_txt(clusters,"agglomerative.txt")

if __name__ == '__main__':
	main()