
import numpy as np
from numpy import pi
from sklearn.metrics import hamming_loss
import matplotlib.pyplot as plt

from scipy.stats import norm


data = np.genfromtxt('wbdc.txt', delimiter=',')

size = [50, 100, 150, 200, 250, 300, 350, 400, 450]
err_trn = []
err_tst = []
for sz in size:
    train, test = data[1:sz+1, :], data[sz+1:, :]
    X_trn, y_trn = train[:, 2:], train[:,1]
    X_tst, y_tst = test[:, 2:], test[:,1]
    
    # Compute the label probabilities
    prob1 = sum(y_trn) / y_trn.size
    prob0 = 1 - prob1
    
    # Compute the class conditional means, and variance for Gaussian Naive Bayes
    # You 'may' obtain four arrays/lists of size 30, two for means for each label, 
    # and two for variances for each label.  
    sum0 = np.zeros([1,30])
    sum1 = np.zeros([1,30])
    varians0 = np.zeros([1, 30])
    varians1 = np.zeros([1, 30])
    count0 = 0;
    count1 = 0;
    for i in range(0, sz):
        if y_trn[i] == 0.:
            count0 = count0 + 1
            sum0 = sum0 + X_trn[i, :]
        else:
            count1 = count1 + 1
            sum1 = sum1 + X_trn[i, :]
    mean0 = sum0 / count0;
    mean1 = sum1 / count1;
    for i in range(0, sz):
        if y_trn[i] == 0.:
            for z in range(0, 30):
                varians0[0, z] = varians0[0, z] + np.square(X_trn[i, z] - mean0[0, z])
        else:
            for z in range(0, 30):
                varians1[0, z] = varians1[0, z] + np.square(X_trn[i, z] - mean1[0, z])
            
    var0 = varians0 / count0;
    var1 = varians1 / count1;
    
    # Compute the training error, and test error of the GNB model you learnt.
    y_trn_result = []
    y_tst_result = []
    for i in range(0, sz): # training data
        sum0_trn = 0
        sum1_trn = 0
        for j in range(0, 30):
            sum0_trn = sum0_trn + np.log((1/np.sqrt((2*pi*var0[0,j])))*np.exp(-np.square(X_trn[i, j]-mean0[0,j])/(2*var0[0,j])))
        for j in range(0, 30):
            sum1_trn = sum1_trn + np.log((1/np.sqrt((2*pi*var1[0,j])))*np.exp(-np.square(X_trn[i, j]-mean1[0,j])/(2*var1[0,j])))
        sum0_trn = sum0_trn + np.log(prob0)
        sum1_trn = sum1_trn + np.log(prob1)
        if sum0_trn > sum1_trn:
            y_trn_result.append(0)
        else:
            y_trn_result.append(1)
    err_trn_cur = sum(np.abs(y_trn_result - y_trn))/y_trn.size 
    err_trn.append(err_trn_cur)
    for i in range(0, y_tst.size): # test data
        sum0_tst = 0
        sum1_tst = 0
        for j in range(0, 30):
            sum0_tst = sum0_tst + np.log(1/np.sqrt((2*pi*var0[0,j]))*np.exp(-np.square(X_tst[i, j]-mean0[0,j])/(2*var0[0,j])))
        for j in range(0, 30):
            sum1_tst = sum1_tst + np.log(1/np.sqrt((2*pi*var1[0,j]))*np.exp(-np.square(X_tst[i, j]-mean1[0,j])/(2*var1[0,j])))
        sum0_tst = sum0_tst + np.log(prob0)
        sum1_tst = sum1_tst + np.log(prob1)
        if sum0_tst > sum1_tst:
            y_tst_result.append(0)
        else:
            y_tst_result.append(1)
    err_tst_cur = sum(np.abs(y_tst_result - y_tst))/y_tst.size 
    err_tst.append(err_tst_cur)
    # You may want to use logarithms for computations
# In a single plot, show the values of training and test 
plt.figure(1,figsize = (12,6))
plt.plot(size, err_trn,'-o',color='red',label = 'Training')
plt.plot(size, err_tst,'-o',color='blue',label = 'Test')
plt.xlabel('size')
plt.ylabel('error rate')
plt.legend(loc='upper right')
plt.title('Gaussian Naive Bayes')
plt.show()
