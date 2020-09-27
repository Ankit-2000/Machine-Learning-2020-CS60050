# part 2 :  Visualization of the fitted curves
# Ankit Bagde
# 17CS30009

import pandas as pd 		
import matplotlib.pyplot as plt 		# to plot graphs 
import numpy as np

# reading train and test files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

x_train = np.array(train['Feature']).reshape(-1,1)
y_train = np.array(train[' Label']).reshape(-1,1)

x_test = np.array(test['Feature']).reshape(-1,1)
y_test = np.array(test[' Label']).reshape(-1,1)


# load required buffers
w_buff = np.load('part1/w_buff.npy')
x_buff = np.load('part1/x_buff.npy')
train_error_buff = np.load('part1/train_error_buff.npy')
test_error_buff = np.load('part1/test_error_buff.npy')
x = np.load('part1/x.npy')


# 2(a) : Plot 9 different curves 

for i in range(1,10):
	print("Plotting graph for n = %d"%(i))
	plt.scatter(x_train, y_train)
	plt.plot(x, ((w_buff[i-1].transpose()).dot(x_buff[i-1])).transpose(), 'g')
	plt.xlabel('Feature')
	plt.ylabel('Label')
	plt.title('Plot : curve for n : %d'%(i))
	plt.savefig('part2/2(a)_graph_n=%d.png'%(i))
	plt.show()

# 2(b) : Report all train/test errors in the form of a plor

print("Plotting graph for all train/test error")
plt.plot([1,2,3,4,5,6,7,8,9], train_error_buff, 'r', Label='Train Error')
plt.plot([1,2,3,4,5,6,7,8,9], test_error_buff, 'g', Label='Test Error')
plt.xlabel('n (from 1 t 9)')
plt.ylabel('Error')
plt.title('Plot : curve for all train/test error')
plt.legend()
plt.savefig('part2/2(b)_train-test_errors.png')
plt.show()

print("All the plots are saved at 'part2' directory")
