# part 1 :  Understanding the data and simple curve fitting
# Ankit Bagde
# 17CS30009

import pandas as pd 		
import matplotlib.pyplot as plt 		# to plot graphs 
import numpy as np

# reading train and test files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train size : "+str(train.shape))
print("Test size  : "+str(test.shape))

x_train = np.array(train['Feature']).reshape(-1,1)
y_train = np.array(train[' Label']).reshape(-1,1)

x_test = np.array(test['Feature']).reshape(-1,1)
y_test = np.array(test[' Label']).reshape(-1,1)

# 1(a) : Plot a feature vs label graph 

# train plot
print("\nPlotting graph :: train data")
plt.plot(x_train, y_train, 'go')
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot : Train data')
plt.savefig('part1/1(a)_train.png')
plt.show()

# test plot
print("Plotting graph :: test data")
plt.plot(x_test, y_test, 'bo', linewidth=1)
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot : Test data')
plt.savefig('part1/1(a)_test.png')
plt.show()

# both train and test plot
print("Plotting graph :: both train and test data")
plt.plot(x_train, y_train, 'go', label='Train')
plt.plot(x_test, y_test, 'bo', label='Test', linewidth=1)
plt.xlabel('Feature')
plt.ylabel('Label')
plt.title('Plot : Both Train & Test data')
plt.legend()
plt.savefig('part1/1(a)_train+test.png')
plt.show()

# 1(b) : curve fitting for n = 1 to 9 


# given learning rate
alpha = 0.05
m = x_train.shape[0]
convergence = 10e-8
print("\nalpha :: %.2f    convergence :: %.7f"%(alpha,convergence))

# buffers to store the data which will be used further
# h_x_buff stores all final predictions of the train data to plot curves in Q.2
h_x_buff = []
w_buff = []
x_buff = []
train_error_buff = []
test_error_buff = []

# for plotting polynomial
x = np.linspace(0, 1, 1000)

for n in range(1,10):
	w = np.zeros((n+1,1))
	phi = np.ones((n+1,x_train.shape[0]))
	temp = np.ones((n+1,x.shape[0]))
	phi_test = np.ones((n+1,x_test.shape[0]))
	
	for i in range(1,n+1):
		phi[i][:] = pow(x_train,i).reshape(x_train.shape[0])
		phi_test[i][:] = pow(x_test,i).reshape(x_test.shape[0])
		temp[i][:] = pow(x,i).reshape(x.shape[0])
	x_buff.append(temp)


	# gradient descent :
	#	example for n=1 
	# w is of shape (2,1)
	# phi is of shape (2,1000)
	# w.T * phi gives (1,1000) - i.e. y_pred.T or h(x).T
	# so, h_x = (w.T * phi).T of shape (1000,1)
	# let d = phi * (h_x - y), which of shape (2,1)
	# so, w = w - alpha*(1/m)*d
	

	it=0
	print("\n------------------------   n = %d   ------------------------"%(n))
	while(1):
		h_x_T = (np.transpose(w)).dot(phi)
		h_x = np.transpose(h_x_T)
		d = phi.dot(h_x - y_train)
		# previous cost
		J_x_prev = np.sum(pow(h_x - y_train,2))/(2*m)
		# gradient descent	
		w = w - alpha*(1/m)*d
		# new cost			
		J_x = (1/(2*m))*np.sum(pow(((np.transpose(w)).dot(phi)).transpose() - y_train , 2))

		# Uncomment this part to see errors after every 5 iterations
		# if((it+1)%5 == 0):
		# 	train_error = J_x
		# 	test_error = (1/(2*x_test.shape[0]))*np.sum(pow(((np.transpose(w)).dot(phi_test)).transpose() - y_test , 2))
		# 	print("itertaion : %d      train_error : %.4f     test_error : %.4f"%(it+1, train_error, test_error))

		if(J_x_prev - J_x <= convergence):
			train_error = J_x
			test_error = (1/(2*x_test.shape[0]))*np.sum(pow(((np.transpose(w)).dot(phi_test)).transpose() - y_test , 2))
			print("Converged :: itertaion : %d      train_error : %.6f     test_error : %.4f"%(it+1, train_error, test_error))
			h_x_buff.append(h_x)
			# h_x_buff[n] = h_x
			w_buff.append(w)
			train_error_buff.append(train_error)
			test_error_buff.append(test_error)
			break;
		it = it+1

# saving all required buffers on disk
print("\nSaving all required buffers on the disk")
np.save('part1/h_x_buff.npy', h_x_buff)
np.save('part1/w_buff.npy', w_buff)
np.save('part1/x_buff.npy', x_buff)
np.save('part1/train_error_buff.npy', train_error_buff)
np.save('part1/test_error_buff.npy', test_error_buff)
np.save('part1/x.npy', x)

print("All the plots and other files saved at 'part1' directory")
