# Ankit Bagde - 17cs30009
# Task 3 - Decision Tree

import pprint
import operator
import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

pd.set_option('mode.chained_assignment', None)			#disble warnings

# using dataset_B
data = pd.read_csv('../data/dataset_B.csv')
print("data size is "+str(data.shape))
attributes = list(data)[:-1]

# Steps
# Calculate entropy of each attribute
# Calculate IG of each attribute
# select max IG and repeat recursively

# calculate a node entropy
def node_entr(unique_val, counts, l):
	node_entropy = 0
	for x in range(len(unique_val)):
		node_entropy += -(counts[x]/l)*np.log2(counts[x]/l) 
	return node_entropy

def attributes_entropy(attributes, node_entropy, data, unique_val):
	info_gain = []
	for i in attributes:
		attribute_vars = np.unique(data[i])
		entropy_sum=0

		# calculating entropy after each split 
		for var in attribute_vars:
			
			# new dataframe as per one particular attribute value
			temp_df = data[data[i] == var]
			unique_val, counts = np.unique(np.array(temp_df[list(temp_df.keys())[-1]]), return_counts=True)

			partition_node_entropy = node_entr(unique_val, counts, len(temp_df[list(temp_df.keys())[-1]]))

			frac = len(temp_df)/len(data)
			entropy_sum += frac*partition_node_entropy 

		info_gain.append(node_entropy - entropy_sum)

	return info_gain


def ID3_tree(data, attributes, tree=None):

	# get unique values and counts of the record
	unique_val, counts = np.unique(np.array(data[list(data.keys())[-1]]), return_counts=True)
	
	if(len(attributes)==0 or data.shape[0] < 10):
		# print("return ")
		target_class = unique_val[np.argmax(counts)]
		return target_class

	node_entropy = node_entr(unique_val, counts, len(data[list(data.keys())[-1]]))

	info_gain = attributes_entropy(attributes, node_entropy, data, unique_val)
	# print(len(info_gain))
	
	node = list(data.keys())[:-1][np.argmax(info_gain)]
	node_vars = np.unique(np.array(data[node]))

	if tree is None:
		tree = {}
		tree[node] = {}

	for value in node_vars:
		prev_class = unique_val[np.argmax(counts)]

		# new data as per partition
		new_data = data[data[node] == value]
		new_data.drop([node], axis=1, inplace=True)
		# print(new_data.shape)

		if(len(new_data[list(new_data.keys())[-1]]) == 0):
			tree[node][value] = prev_class
			# print("no attr")
			continue

		new_unique_val, new_counts = np.unique(np.array(new_data[list(new_data.keys())[-1]]), return_counts=True)

		if(len(new_counts) == 1):
			# print("unique")
			tree[node][value] = new_unique_val[0]
		else:
			# print("aage")
			tree[node][value] = ID3_tree(new_data, list(new_data.keys())[:-1]) 

	return tree

def Test_tree(tree, test_data):
    y_pred = []
    y_true = list(test_data['quality'])

    for i in range(len(test_data)):
        node = tree
        while(True):
        	# check if node is dict
            if isinstance(node, dict):
                for key in node.keys():
                    node_val = key
                if test_data[node_val][i] in node[node_val]:
                    node = node[node_val][test_data[node_val][i]]
                else:
                    for j in node[node_val]:
                        val = j
                    node = node[node_val][val]
            else:
                break
        y_pred.append(node)

    return (accuracy_score(y_true,y_pred), precision_score(y_true, y_pred, zero_division=1, average='macro'), recall_score(y_true, y_pred, average='macro'))

def Sci_kit_DT(x, y, test_x, y_true):
	model = DecisionTreeClassifier(criterion='entropy', min_samples_split=10).fit(x,y)	
	y_pred = model.predict(test_x)
	
	return (accuracy_score(y_true,y_pred), precision_score(y_true, y_pred, zero_division=1, average='macro'), recall_score(y_true, y_pred, average='macro'))


print("\n------ Fold - 1 -------\n")
train_1 = data[:int(data.shape[0]*2/3)].reset_index()
cv_1 = data[int(data.shape[0]*2/3):].reset_index()
train_1.drop(['index'], axis=1, inplace=True)
cv_1.drop(['index'], axis=1, inplace=True)

tree = ID3_tree(train_1, attributes)
tuple_1 = Test_tree(tree, cv_1)
tuple_2 = Sci_kit_DT(train_1.values[:,:-1], train_1.values[:,-1], cv_1.values[:,:-1], cv_1.values[:,-1])

print("Our implementation of Decision Tree  ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_1[0], tuple_1[1], tuple_1[2]))
print("Sci-kit learn implementation         ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_2[0], tuple_2[1], tuple_2[2]))

print("\n------ Fold - 2 -------\n")
train_2 = pd.concat([data[:int(data.shape[0]*1/3)],data[int(data.shape[0]*2/3):]]).reset_index()
cv_2 = data[int(data.shape[0]*1/3):int(data.shape[0]*2/3)].reset_index()
train_2.drop(['index'], axis=1, inplace=True)
cv_2.drop(['index'], axis=1, inplace=True)

tree = None
tree = ID3_tree(train_2, attributes)
tuple_t = Test_tree(tree, cv_2)
print("Our implementation of Decision Tree  ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_t[0], tuple_t[1], tuple_t[2]))
tuple_1 = tuple(map(operator.add, tuple_1, tuple_t))

tuple_t = Sci_kit_DT(train_2.values[:,:-1], train_2.values[:,-1], cv_2.values[:,:-1], cv_2.values[:,-1])
print("Sci-kit learn implementation         ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_t[0], tuple_t[1], tuple_t[2]))
tuple_2 = tuple(map(operator.add, tuple_2, tuple_t))


print("\n------ Fold - 3 -------\n")
train_3 = data[int(data.shape[0]*1/3):].reset_index()
cv_3= data[:int(data.shape[0]*1/3)].reset_index()
train_3.drop(['index'], axis=1, inplace=True)
cv_3.drop(['index'], axis=1, inplace=True)

tree = None
tree = ID3_tree(train_3, attributes)
tuple_t = Test_tree(tree, cv_3)
print("Our implementation of Decision Tree  ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_t[0], tuple_t[1], tuple_t[2]))
tuple_1 = tuple(map(operator.add, tuple_1, tuple_t))

tuple_t = Sci_kit_DT(train_3.values[:,:-1], train_3.values[:,-1], cv_3.values[:,:-1], cv_3.values[:,-1])
print("Sci-kit learn implementation         ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_t[0], tuple_t[1], tuple_t[2]))
tuple_2 = tuple(map(operator.add, tuple_2, tuple_t))

# Average of all accuracy, precision, recall
tuple_1 = tuple(x/3 for x in tuple_1)
tuple_2 = tuple(x/3 for x in tuple_2)

print("\n\n------ Average -------\n")
print("Our implementation of Decision Tree  ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_1[0], tuple_1[1], tuple_1[2]))
print("Sci-kit learn implementation         ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_2[0], tuple_2[1], tuple_2[2]))

