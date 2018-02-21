import numpy as np
from sklearn.metrics import hamming_loss
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

data = np.genfromtxt('wbdc.txt', delimiter=',') # wdbc is loaded into an array 'data'

# PART 3: We will use minimum leaf size, which ensures that each node must have at least a certain number of samples
size = [50, 200, 400]
minimum_leaf_size = range(1, 16)
i = 0
for sz in size:
    train, test = data[1:sz+1, :], data[sz+1:, :] # divide data into training and test sets
    X_trn, y_trn = train[:, 2:], train[:,1] # separate the features and labels of training
    X_tst, y_tst = test[:, 2:], test[:,1] # separate the features and labels of test
    errTest = []
    errTrn = []
    for minimum in minimum_leaf_size:
        clf = tree.DecisionTreeClassifier(min_samples_leaf = minimum, criterion ='entropy',random_state = 0)
        clf = clf.fit(X_trn, y_trn)
        predictTest = clf.predict(X_tst)
        errorTest = predictTest - y_tst
        errorTest = abs(errorTest)
        curSumTest = sum(errorTest)
        curErrTest = curSumTest / errorTest.size
        errTest.append(curErrTest)
        predictTrn = clf.predict(X_trn)
        errorTrn = predictTrn - y_trn
        errorTrn = abs(errorTrn)
        curSumTrn = sum(errorTrn)
        curErrTrn = curSumTrn / errorTrn.size
        errTrn.append(curErrTrn)
    plt.figure(i,figsize = (12,6))
    plt.plot(minimum_leaf_size, errTrn,color='red',label = 'Training')
    plt.plot(minimum_leaf_size, errTest,color='blue',label = 'Test') 
    plt.xlabel('minimum_leaf_size')
    plt.ylabel('error')
    plt.legend(loc='upper right')
    plt.title('Decision Tree of size %d, Entropy Criterion' %(sz))
    plt.show()
    i = i + 1
