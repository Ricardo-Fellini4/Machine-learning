from __future__ import division
import numpy as np 
from mnist import MNIST
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.metrics import accuracy_score

import time

# you need to download the MNIST dataset first
# at: http://yann.lecun.com/exdb/mnist/
mndata = MNIST('D:\Machine learning\MNIST') # path to your MNIST folder 
mndata.load_testing()
mndata.load_training()
X_test = mndata.test_images
X_train = mndata.train_images
y_test = np.asarray(mndata.test_labels)
y_train = np.asarray(mndata.train_labels)

# uncomment then run 
start_time = time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_time = time.time()
print( "Accuracy of 1NN for MNISST: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
print ("Running time: %.2f (s)" % (end_time - start_time))