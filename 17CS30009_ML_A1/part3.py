# part 3 :  Regularization
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

alpha = 0.05
m = x_train.shape[0]
convergence = 10e-8

# load required buffers
w_buff = np.load('part1/w_buff.npy')
x_buff = np.load('part1/x_buff.npy')
train_error_buff = np.load('part1/train_error_buff.npy').tolist()
test_error_buff = np.load('part1/test_error_buff.npy').tolist()
x = np.load('part1/x.npy')

# buffers required for plot
lasso_buff_train = []
lasso_buff_test = []
ridge_buff_train = []
ridge_buff_test = []

# possible values for lambda taken into consideration
lambda_reg = [0.25, 0.5, 0.75, 1]

# get n for minimum and maximum train errors
min_n = train_error_buff.index(min(train_error_buff)) + 1
max_n = train_error_buff.index(max(train_error_buff)) + 1

print("\nMinimum training error obtained for n = %d"%(min_n))
print("Maximum training error obtained for n = %d"%(max_n))

# we need to do regularization on req_n
req_n = [min_n, max_n]

# 3(a) : Lasso Regression

print("\n----------------------------------------------------------------")
print("-                        Lasso Regression                      -")
print("----------------------------------------------------------------")
for n in req_n:
	print("\n----------------------------  n = %d  --------------------------"%(n))
	for l in lambda_reg:

		w = np.zeros((n+1,1))
		phi = np.ones((n+1,x_train.shape[0]))
		phi_test = np.ones((n+1,x_test.shape[0]))

		for i in range(1,n+1):
			phi[i][:] = pow(x_train,i).reshape(x_train.shape[0])
			phi_test[i][:] = pow(x_test,i).reshape(x_test.shape[0])

		# gradient descent for lasso regression:
		#	example for n=1 
		# w is of shape (2,1)
		# phi is of shape (2,1000)
		# w.T * phi gives (1,1000) - i.e. y_pred.T or h(x).T
		# so, h_x = (w.T * phi).T of shape (1000,1)
		# let d = phi * (h_x - y), which of shape (2,1)
		
		# w = w - alpha*(1/m)*d                   		for i=0
		# w = w - alpha*lambda/(2*m) - alpha*(1/m)*d 	for i>=1	

		it=0
		while(1):
			h_x_T = (np.transpose(w)).dot(phi)
			h_x = np.transpose(h_x_T)
			d = phi.dot(h_x - y_train)
			# previous cost
			J_x_prev = (np.sum(pow(h_x - y_train,2)) + l*np.sum(w[1:]))/(2*m)
			# gradient descent
			w[0] = w[0] - alpha*(1/m)*d[0]	
			w[1:] = w[1:] - (alpha*l/(2*m)) - alpha*(1/m)*d[1:]
			# new cost			
			J_x = (1/(2*m))*(np.sum(pow(((np.transpose(w)).dot(phi)).transpose() - y_train , 2)) + l*np.sum(w[1:]))

			# Uncomment this part to see errors after every 5 iterations
			# if((it+1)%5 == 0):
			# 	train_error = J_x
			# 	test_error = (1/(2*x_test.shape[0]))*np.sum(pow(((np.transpose(w)).dot(phi_test)).transpose() - y_test , 2))
			# 	print("itertaion : %d      train_error : %.4f     test_error : %.4f"%(it+1, train_error, test_error))

			if(J_x_prev - J_x <= convergence):
				train_error = J_x
				test_error = (1/(2*x_test.shape[0]))*(np.sum(pow(((np.transpose(w)).dot(phi_test)).transpose() - y_test , 2)) + l*np.sum(w[1:]))
				print("Converged ::   lambda : %.2f      itertaion : %d      train_error : %.6f      test_error : %f"%(l,it+1, train_error, test_error))
				lasso_buff_test.append(test_error)
				lasso_buff_train.append(train_error)
				break;
			it = it+1


# 3(b) : Ridge Regression

print("\n\n----------------------------------------------------------------")
print("-                        Ridge Regression                      -")
print("----------------------------------------------------------------")

for n in req_n:
	print("\n------------------------ n = %d ------------------------"%(n))
	for l in lambda_reg:
		w = np.zeros((n+1,1))
		phi = np.ones((n+1,x_train.shape[0]))
		phi_test = np.ones((n+1,x_test.shape[0]))

		for i in range(1,n+1):
			phi[i][:] = pow(x_train,i).reshape(x_train.shape[0])
			phi_test[i][:] = pow(x_test,i).reshape(x_test.shape[0])

		# gradient descent for ridge regression:
		#	example for n=1 
		# w is of shape (2,1)
		# phi is of shape (2,1000)
		# w.T * phi gives (1,1000) - i.e. y_pred.T or h(x).T
		# so, h_x = (w.T * phi).T of shape (1000,1)
		# let d = phi * (h_x - y), which of shape (2,1)

		# w = w - alpha*(1/m)*d                   	for i=0
		# w = w(1-alpha*lambda/m) - alpha*(1/m)*d 	for i>=1
		

		it=0
		while(1):
			h_x_T = (np.transpose(w)).dot(phi)
			h_x = np.transpose(h_x_T)
			d = phi.dot(h_x - y_train)
			# previous cost
			J_x_prev = (np.sum(pow(h_x - y_train,2)) + l*np.sum(pow(w[1:],2)))/(2*m)
			# gradient descent
			w[0] = w[0] - alpha*(1/m)*d[0]	
			w[1:] = w[1:]*(1-(alpha*l/m)) - alpha*(1/m)*d[1:]
			# new cost			
			J_x = (1/(2*m))*(np.sum(pow(((np.transpose(w)).dot(phi)).transpose() - y_train , 2)) + l*np.sum(pow(w[1:],2)))

			# Uncomment this part to see errors after every 5 iterations
			# if((it+1)%5 == 0):
			# 	train_error = J_x
			# 	test_error = (1/(2*x_test.shape[0]))*np.sum(pow(((np.transpose(w)).dot(phi_test)).transpose() - y_test , 2))
			# 	print("itertaion : %d      train_error : %.4f     test_error : %.4f"%(it+1, train_error, test_error))

			if(J_x_prev - J_x <= convergence):
				train_error = J_x
				test_error = (1/(2*x_test.shape[0]))*(np.sum(pow(((np.transpose(w)).dot(phi_test)).transpose() - y_test , 2)) + l*np.sum(pow(w[1:],2)))
				print("Converged ::   lambda : %.2f      itertaion : %d      train_error : %.6f      test_error : %.4f"%(l,it+1, train_error, test_error))
				ridge_buff_test.append(test_error)
				ridge_buff_train.append(train_error)
				break;
			it = it+1


# plot the train & test error vs lambda
print("Plotting train_error graph for n = %d"%(min_n))
plt.plot(lambda_reg, lasso_buff_train[:4], 'rs', Label='Train Error - Lasso')
plt.plot(lambda_reg, ridge_buff_train[:4], 'b^', Label='Train Error - ridge')
plt.plot([0.25,1], [train_error_buff[min_n-1], train_error_buff[min_n-1]], 'g--', Label='Train Error - without regularization')
plt.xlabel('lambda')
plt.ylabel('Train Error')
plt.title('Plot : curve for Train error (minimum train error n=%d)'%(min_n))
plt.legend()
plt.savefig('part3/min_n_train_error.png')
plt.show()


print("Plotting test_error graph for n = %d"%(min_n))
plt.plot(lambda_reg, lasso_buff_test[:4], 'rs', Label='Test Error - Lasso')
plt.plot(lambda_reg, ridge_buff_test[:4], 'b^', Label='Test Error - ridge')
plt.plot([0.25,1], [test_error_buff[min_n-1], test_error_buff[min_n-1]], 'g--', Label='Test Error - without regularization')
plt.xlabel('lambda')
plt.ylabel('Test Error')
plt.title('Plot : curve for Test error (minimum train error n=%d)'%(min_n))
plt.legend()
plt.savefig('part3/min_n_test_error.png')
plt.show()


print("Plotting train_error graph for n = %d"%(max_n))
plt.plot(lambda_reg, lasso_buff_train[4:], 'rs', Label='Train Error - Lasso')
plt.plot(lambda_reg, ridge_buff_train[4:], 'b^', Label='Train Error - ridge')
plt.plot([0.25,1], [train_error_buff[max_n-1], train_error_buff[max_n-1]], 'g--', Label='Train Error - without regularization')
plt.xlabel('lambda')
plt.ylabel('Train Error')
plt.title('Plot : curve for Train error (for maximum train error n=%d)'%(max_n))
plt.legend()
plt.savefig('part3/max_n_train_error.png')
plt.show()


print("Plotting test_error graph for n = %d"%(max_n))
plt.plot(lambda_reg, lasso_buff_test[4:], 'rs', Label='Test Error - Lasso')
plt.plot(lambda_reg, ridge_buff_test[4:], 'b^', Label='Test Error - ridge')
plt.plot([0.25,1], [test_error_buff[max_n-1], test_error_buff[max_n-1]], 'g--', Label='Test Error - without regularization')
plt.xlabel('lambda')
plt.ylabel('Test Error')
plt.title('Plot : curve for Test error (maximum train error n=%d)'%(max_n))
plt.legend()
plt.savefig('part3/max_n_test_error.png')
plt.show()

print("All the plots saved at 'part3' directory")
