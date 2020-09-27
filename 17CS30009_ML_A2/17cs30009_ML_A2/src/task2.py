# Ankit Bagde - 17cs30009
# Task 2 - Logistic Regression

import pandas as pd
import numpy as np
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


# using dataset_A for this part
data = pd.read_csv('../data/dataset_A.csv')
print("data size is : "+str(data.shape))
data = data.values								# converting to np ndarray

# Implemetation of LR
def sigmoid(x):
	return 1/(1+np.exp(-x))

def hypothesis(x,theta):
	return sigmoid(x.dot(theta))

def cost(h,y):
	return -(1/y.shape[0])*(np.log(h.T).dot(y) + np.log((1-h).T).dot(1-y))

def Logistic_Reg(x, y, test_x, test_y):
	alpha = 0.05
	convergence = 1e-6
	m = y.shape[0]

	theta = np.zeros((x.shape[1]+1,1))
	x = np.concatenate((np.ones((x.shape[0],1),dtype=int),x),axis=1)
	h = hypothesis(x,theta)		
	prev_cost = cost(h,y)

	it=0
	while(1):

		d = x.T.dot(h-y)
		theta = theta - (alpha/m)*d

		h = hypothesis(x,theta)
		new_cost = cost(h,y)

		if(prev_cost - new_cost <= convergence): 
			y_pred = [0 if x<0.5 else 1 for x in h]
			# print("Our implementation of LR     ->  train_accuracy_score  ::  %.5f "%(accuracy_score(y,y_pred), it+1))
			
			test_x = np.concatenate((np.ones((test_x.shape[0],1),dtype=int),test_x),axis=1)
			test_h = hypothesis(test_x,theta)
			y_pred = [0 if i<0.5 else 1 for i in test_h]

			return (accuracy_score(test_y,y_pred), precision_score(test_y, y_pred, zero_division=1), recall_score(test_y, y_pred))

		prev_cost = new_cost
		it = it+1

# Implementation using scikit-learn package
def Sci_kit_LR(x, y, test_x, test_y):
	model = LogisticRegression(penalty='none', solver='saga', tol=1e-6, max_iter=13000).fit(x,y.reshape(-1,))
	# print("Sci-kit learn implementation ->  train_accuracy_score  ::  %.5f"%(accuracy_score(y, model.predict(x))))
	
	y_pred = model.predict(test_x)
	return (accuracy_score(test_y,y_pred), precision_score(test_y, y_pred, zero_division=1), recall_score(test_y, y_pred))


# # 3-folds cv

print("\n------ Fold - 1 -------\n")
train_1_x = data[:int(data.shape[0]*2/3),:11]
train_1_y = data[:int(data.shape[0]*2/3),11].reshape(-1,1)
cv_1_x = data[int(data.shape[0]*2/3):,:11]
cv_1_y = data[int(data.shape[0]*2/3):,11].reshape(-1,1)

tuple_1 = Logistic_Reg(train_1_x, train_1_y, cv_1_x, cv_1_y)
tuple_2 = Sci_kit_LR(train_1_x, train_1_y, cv_1_x, cv_1_y)

print("Our implementation of LR      ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_1[0], tuple_1[1], tuple_1[2]))
print("Sci-kit learn implementation  ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_2[0], tuple_2[1], tuple_2[2]))


print("\n------ Fold - 2 -------\n")
train_2_x = np.concatenate((data[:int(data.shape[0]*1/3),:11],data[int(data.shape[0]*2/3):,:11]))
train_2_y = np.concatenate((data[:int(data.shape[0]*1/3),11],data[int(data.shape[0]*2/3):,11])).reshape(-1,1)
cv_2_x = data[int(data.shape[0]*1/3):int(data.shape[0]*2/3),:11]
cv_2_y = data[int(data.shape[0]*1/3):int(data.shape[0]*2/3), 11].reshape(-1,1)

tuple_t = Logistic_Reg(train_2_x, train_2_y, cv_2_x, cv_2_y)
print("Our implementation of LR      ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_t[0], tuple_t[1], tuple_t[2]))
tuple_1 = tuple(map(operator.add, tuple_1, tuple_t))

tuple_t = Sci_kit_LR(train_2_x, train_2_y, cv_2_x, cv_2_y)
print("Sci-kit learn implementation  ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_t[0], tuple_t[1], tuple_t[2]))
tuple_2 = tuple(map(operator.add, tuple_2, tuple_t))


print("\n------ Fold - 3 -------\n")
train_3_x = data[int(data.shape[0]*1/3):,:11]
train_3_y = data[int(data.shape[0]*1/3):,11].reshape(-1,1)
cv_3_x = data[:int(data.shape[0]*1/3),:11]
cv_3_y = data[:int(data.shape[0]*1/3), 11].reshape(-1,1)

tuple_t = Logistic_Reg(train_3_x, train_3_y, cv_3_x, cv_3_y)
print("Our implementation of LR      ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_t[0], tuple_t[1], tuple_t[2]))
tuple_1 = tuple(map(operator.add, tuple_1, tuple_t))

tuple_t = Sci_kit_LR(train_3_x, train_3_y, cv_3_x, cv_3_y)
print("Sci-kit learn implementation  ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_t[0], tuple_t[1], tuple_t[2]))
tuple_2 = tuple(map(operator.add, tuple_2, tuple_t))

# Average of all accuracy, precision, recall
tuple_1 = tuple(x/3 for x in tuple_1)
tuple_2 = tuple(x/3 for x in tuple_2)

print("\n\n------ Average -------\n")
print("Our implementation of LR      ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_1[0], tuple_1[1], tuple_1[2]))
print("Sci-kit learn implementation  ->  Accuracy  ::  %.5f    precision  ::  %.5f    Recall  ::  %.5f\n"%(tuple_2[0], tuple_2[1], tuple_2[2]))



