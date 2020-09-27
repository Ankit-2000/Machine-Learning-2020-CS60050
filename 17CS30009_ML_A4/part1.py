import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# preprocess the data, return the train and test set
def preprocess():
	data = np.loadtxt(fname="data/seeds_dataset.txt")
	labels = data[:,7].reshape(-1,1)
	data = np.delete(data, 7, 1)

	# z-normalization
	scaler = StandardScaler()
	scaled_data = scaler.fit_transform(data)

	# one hot encoding
	encoder = OneHotEncoder()
	labels = encoder.fit_transform(labels).toarray()

	# test-train split
	x_train, x_test, y_train, y_test = train_test_split(scaled_data, labels, test_size=0.2)

	# save train and test data for B part
	np.save('data/x_train.npy', x_train)
	np.save('data/x_test.npy', x_test)
	np.save('data/y_train.npy', y_train)
	np.save('data/y_test.npy', y_test)

	return x_train, x_test, y_train, y_test

# create minibatches of 32 examples each
def data_loader(x_train, y_train):
	x_batch = []
	y_batch = []
	n_batches = x_train.shape[0]//32						
	for i in range(n_batches):
		x_batch.append(x_train[i*32:(i+1)*32,:])
		y_batch.append(y_train[i*32:(i+1)*32])

	# remaining examples in last batch
	x_batch.append(x_train[n_batches*32:,:])
	y_batch.append(y_train[n_batches*32:])

	return x_batch, y_batch

# initialize the weights and return a dict cotaining all weights and bias matrices of all layers 
def weight_intializer(layer_dims):
	# W[l] - shape is no. of nodes in lth layer x no. of nodes in l-1 layer
	parameters = {}
	for i in range(1,len(layer_dims)):
		parameters["W"+str(i)] = np.random.uniform(low = -1, high = 1, size=(layer_dims[i], layer_dims[i-1])) 
		parameters["b"+str(i)] = np.zeros((layer_dims[i],1))
	return parameters

# stable softmax definition
def softmax(x):
	exp_x = np.exp(x - np.max(x, axis=1).reshape(-1,1))
	return exp_x/(np.sum(exp_x, axis=1).reshape(-1,1))

def relu(x):
	return np.maximum(0,x)

def derivative_relu(Z):
	x = (Z>0).astype(int)
	return x

def sigmoid(x):
	return 1/(1+np.exp(-x))

def derivative_sigmoid(Z):
	return sigmoid(Z)*(1-sigmoid(Z))

# categorical cross-entropy loss - y is one hot encoded (m,classes)
def cost_function(y_hat, y):
	return -np.mean(y*np.log(y_hat+1e-9))

# returns accuracy for the given data
def accuracy(parameters, x, y, activation_fns):
	y_hat, _ = forward(x, parameters, activation_fns)
	pred_label = np.argmax(y_hat, axis=1)
	y_label = np.argmax(y, axis=1)
	p = np.equal(pred_label, y_label)
	return np.sum(p.astype(int))/x.shape[0]

# print output as per the given format
def print_output(part, parameters, x_train, x_test, y_train, y_test, activation_fns, train_acc, test_acc):
	print("Part "+part+" :")
	print("  Final train accuracy is %f"%(accuracy(parameters, x_train, y_train, activation_fns)))
	print("  Final test  accuracy is %f"%(accuracy(parameters, x_test, y_test, activation_fns)))
	
	x_axis = [i*10 for i in range(1,21)]
	plt.plot(x_axis, train_acc, 'r', Label="Train accuracy")
	plt.plot(x_axis, test_acc, 'g', Label="Test accuracy")
	plt.legend(loc=4)
	plt.title("Part "+part)
	plt.xlabel('epochs')
	plt.ylabel('Accuracy')
	plt.savefig(part+'.png')
	plt.show()

# defines a forward pass on the neural network
def forward(x, parameters, activation_fns):
	A=x
	n_layers = (int)(len(parameters)/2)
	cache = []
	
	# for layers 1 to second last layer
	for l in range(1,n_layers):
		Z = np.dot(A, np.transpose(parameters["W"+str(l)])) + np.transpose(parameters["b"+str(l)])
		if(activation_fns[l-1] == "relu"):
			A_next  = relu(Z)
		elif(activation_fns[l-1] == "sigmoid"):
			A_next = sigmoid(Z)
		cache.append((A, Z, parameters["W"+str(l)], parameters["b"+str(l)]))
		A = A_next
	
	# for last layer
	Z = np.dot(A, np.transpose(parameters["W"+str(n_layers)])) + np.transpose(parameters["b"+str(n_layers)])
	A_next = softmax(Z)

	cache.append((A, Z, parameters["W"+str(n_layers)], parameters["b"+str(n_layers)]))
	return A_next, cache

# defines a back propogation on the neural network
def Backpropogation(y_hat, y, cache, activation_fns):

	gradients = {}
	n_layers = len(cache)

	# As we are using softmax, we can directly compute dL/dZ,
	# instead of (dL/dy_hat * dy_hat/dZ)
	# https://deepnotes.io/softmax-crossentropy

	# for the last layer (linear  ->  softmax)
	A_l_1, Z, W, b= cache[n_layers-1]
	m = A_l_1.shape[0]

	# softmax:
	dZ  = y_hat - y

	# linear:
	gradients["dW" + str(n_layers)] = (1/m)*(np.dot(dZ.T, A_l_1))							 # dW				
	gradients["db" + str(n_layers)] = (1/m)*np.sum(dZ, axis=0, keepdims=True).reshape(-1,1)	 # db
	gradients["dA" + str(n_layers-1)] = dZ.dot(W)											 # dA_(l-1)

	# for the layers (n_layers-2) to 1
	for i in reversed(range(n_layers-1)):

		# for the layer (linear  ->  activation fn)
		A_prev, Z, W, b = cache[i]
		# print(A_prev.shape)
		m = A_prev.shape[0]
		dA = gradients["dA"+str(i+1)]

		# activation function:
		if(activation_fns[i] == "relu"):
			dZ = dA*derivative_relu(Z)
		elif(activation_fns[i] == "sigmoid"):
			dZ = dA*derivative_sigmoid(Z)
		
		# linear:
		gradients["dW" + str(i+1)] = (1/m)*(np.dot(dZ.T, A_prev))							# dW
		gradients["db" + str(i+1)] = (1/m)*np.sum(dZ, axis=0, keepdims=True).reshape(-1,1)	# db
		gradients["dA" + str(i)] = np.dot(dZ,W) 											# dA_i	

	return gradients

# update parameters after calculating gradients
def update_parameters(alpha, gradients, parameters):
	n_layers = len(parameters)//2
	for i in range(n_layers):
		parameters["W"+str(i+1)] -= alpha*gradients["dW"+str(i+1)] 
		parameters["b"+str(i+1)] -= alpha*gradients["db"+str(i+1)]
	return parameters

# Trains the neural network
def training(x, y, x_train, y_train, x_test, y_test, layer_dims, activation_fns, parameters, epochs, alpha):
	
	n_batches = len(x)
	train_acc = []
	test_acc = []

	# for number of epochs
	for i in range(epochs):
		cost=0
		# for all mini batches
		for j in range(n_batches):
		
			# forward propogation
			y_hat, cache = forward(x[j], parameters, activation_fns)
		
			# compute cost
			cost += cost_function(y_hat, y[j])

			# back propogation
			gradients = Backpropogation(y_hat, y[j], cache, activation_fns)

			# update parameters
			parameters = update_parameters(alpha, gradients, parameters)

		# save data for plot
		if((i+1)%10 == 0):
			train_acc.append(accuracy(parameters, x_train, y_train, activation_fns))
			test_acc.append(accuracy(parameters, x_test, y_test, activation_fns))
			# print("Train accuracy after %d epoch is %f and loss is %f "%(i+1, train_acc[-1], cost))
			# print("Test accuracy after %d epoch is %f"%(i+1, test_acc[-1]))

	return parameters, train_acc, test_acc

def main():
	
	# preprocess
	x_train, x_test, y_train, y_test = preprocess()
	print("Train data shape is : %s"%(str(x_train.shape)))
	print("Test data shape is : %s"%(str(x_test.shape)))


	# create minibatches with 32 examples each
	x_batch, y_batch = data_loader(x_train, y_train)


	###################### 1A Neural Network ################################# 
	
	# layers - (input, hidden, output)
	layer_dims = [x_train.shape[1],32,3]
	activation_fns = ["sigmoid"]
	epochs = 200
	alpha = 0.01

	# initialize weights
	parameters = weight_intializer(layer_dims)

	# begin training
	parameters, train_acc, test_acc = training(	x_batch, y_batch,
												x_train, y_train,
												x_test, y_test,
												layer_dims, activation_fns, parameters,
												epochs, alpha
											)
	print_output("1A", parameters, x_train, x_test, y_train, y_test, activation_fns, train_acc, test_acc)


	##################### 1B Neural Network ################################# 
	
	# layers - (input, hidden, output)
	layer_dims = [x_train.shape[1], 64, 32, 3]
	activation_fns = ["relu", "relu"]
	epochs = 200
	alpha = 0.01

	# initialize weights
	parameters = weight_intializer(layer_dims)

	# begin training
	parameters, train_acc, test_acc = training(	x_batch, y_batch,
												x_train, y_train,
												x_test, y_test,
												layer_dims, activation_fns, parameters,
												epochs, alpha
											)
	print_output("1B", parameters, x_train, x_test, y_train, y_test, activation_fns, train_acc, test_acc)


if __name__ == "__main__":
	main()