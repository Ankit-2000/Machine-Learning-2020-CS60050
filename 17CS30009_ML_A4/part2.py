import warnings
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

# ignore convergence warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

# load dataset
x_train = np.load('data/x_train.npy')
x_test = np.load('data/x_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

print("Part 2 Specification 1A :")
model = MLPClassifier ( hidden_layer_sizes=(32,), activation='logistic',
						solver='sgd', learning_rate='constant', 
						learning_rate_init=0.01, batch_size=32, 
						max_iter=200)
model.fit(x_train, y_train)
print("  Final train accuracy is %f"%(model.score(x_train, y_train)))
print("  Final test  accuracy is %f\n"%(model.score(x_test, y_test)))
	

print("Part 2B Specification 1B :")
model = MLPClassifier ( hidden_layer_sizes=(64,32,), activation='relu', 
						solver='sgd', learning_rate='constant', 
						learning_rate_init=0.01, batch_size=32, 
						max_iter=200)
model.fit(x_train, y_train)
print("  Final train accuracy is %f"%(model.score(x_train, y_train)))
print("  Final test  accuracy is %f\n"%(model.score(x_test, y_test)))
