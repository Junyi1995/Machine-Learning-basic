
import numpy as np
from sklearn.metrics import hamming_loss
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Feel free to use other libraries.

data = np.genfromtxt('wbdc.txt', delimiter=',') # wdbc is loaded into an array 'data'

size = [200, 400]
max_depth = range(2, 13)
errTest = []
errTrn =[]
plt.figure(1,figsize = (12,6))
sz = 200
#for sz in size:
train, test = data[1:sz+1, :], data[sz+1:, :] # divide data into training and test sets
X_trn, y_trn = train[:, 2:], train[:,1] # separate the features and labels of training
X_tst, y_tst = test[:, 2:], test[:,1] # separate the features and labels of test
for x in max_depth:
    clf = tree.DecisionTreeClassifier(max_depth = x, criterion ='entropy')
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

plt.figure(1,figsize = (12,6))
plt.plot(max_depth, errTrn,'-o',color='red',label = 'Training')
plt.plot(max_depth, errTest,'-o',color='blue',label = 'Test')
plt.xlabel('depth')
plt.ylabel('error')
plt.legend(loc='upper right')
plt.title('Decision Tree of size %d, Entropy Criterion' %(sz))
plt.show()
