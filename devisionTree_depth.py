import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Feel free to use other libraries.

data = np.genfromtxt('wbdc.txt', delimiter=',') # wdbc is loaded into an array 'data'


# PART ONE:
i = 0;
errTest = []
errTrn = []
errTest3 = []
errTrn3 = []
size = [50, 100, 150, 200, 250, 300, 350, 400, 450] 
for sz in size:
    train, test = data[1:sz+1, :], data[sz+1:, :] # divide data into training and test sets
    X_trn, y_trn = train[:, 2:], train[:,1] # separate the features and labels of training
    X_tst, y_tst = test[:, 2:], test[:,1] # separate the features and labels of test
    
    # Compute the training and test errors for depth = 2, and plot them as a function of sz
    # Compute the training and test errors for depth = 3, and plot them as a function of sz
    
    # CODE GOES HERE (YOU CAN USE THE SKLEARN DECISION TREES)
    # Train
    clf = tree.DecisionTreeClassifier(max_depth = 2,random_state = 0)
    clf = clf.fit(X_trn, y_trn)
    clf3 = tree.DecisionTreeClassifier(max_depth = 3,random_state = 0)
    clf3 = clf3.fit(X_trn, y_trn)
    predictTest = clf.predict(X_tst)
    predictTest3 = clf3.predict(X_tst)
    errorTest = predictTest - y_tst
    errorTest3 = predictTest3 - y_tst
    errorTest = abs(errorTest)
    curSumTest = sum(errorTest)
    errorTest3 = abs(errorTest3)
    curSumTest3 = sum(errorTest3)
    curErrTest = curSumTest / errorTest.size
    curErrTest3 = curSumTest3 / errorTest3.size
    errTest.append(curErrTest)
    errTest3.append(curErrTest3)
    predictTrn = clf.predict(X_trn)
    predictTrn3 = clf3.predict(X_trn)
    errorTrn = predictTrn - y_trn
    errorTrn = abs(errorTrn)
    curSumTrn = sum(errorTrn)
    curErrTrn = curSumTrn / predictTrn.size
    errTrn.append(curErrTrn)
    errorTrn3 = predictTrn3 - y_trn
    errorTrn3 = abs(errorTrn3)
    curSumTrn3 = sum(errorTrn3)
    curErrTrn3 = curSumTrn3 / predictTrn3.size
    errTrn3.append(curErrTrn3)



plt.figure(1,figsize = (12,6))
plt.plot(size, errTrn,'-o',color='red',label = 'Training')
plt.plot(size, errTest,'-o',color='blue',label = 'Test')
#plt.figure(figsize=(12,1)) 
plt.xlabel('size')
plt.ylabel('error')
plt.legend(loc='upper right')
plt.title('Decision Tree of Depth 2')
plt.show()
 
plt.figure(2,figsize = (12,6))
plt.plot(size, errTrn3,'-o',color='red',label = 'Training')
plt.plot(size, errTest3,'-o',color='blue',label = 'Test') 
plt.xlabel('size')
plt.ylabel('error')
plt.legend(loc='upper right')
plt.title('Decision Tree of Depth 3')
plt.show()
