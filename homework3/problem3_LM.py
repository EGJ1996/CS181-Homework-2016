# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
import random as random
import time
from sklearn.cross_validation import train_test_split
from Perceptron import Perceptron

# Implement this class
class KernelPerceptron(Perceptron):
    def __init__(self, numsamples):
        self.numsamples = numsamples

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        assert(X.shape[0] == Y.shape[0])
        
        sample_indices = range(X.shape[0]) # for use in random sampling
        A = np.zeros(X.shape[0]) # create initial set of alpha values
        S = set() # create initial set S of support vectors
        
        for sim in range(self.numsamples):
            t = random.sample(sample_indices, 1) # pick random sample from training data and write x, y, and t
            x_t = X[t] # shape is (1,2)
            y_t = Y[t]
            a_t = A[t]

            yhat_t = 0 # initialize yhat_t to 0 to begin the sum
            for i in S: # sum over support vectors to update yhat_t
                yhat_t += A[i]*np.dot(x_t,X[i]) # note that i[2] is alpha_i and i[0] is x_i

            if y_t*yhat_t <= 0: # check condition. if true update set of support vectors and a_t
                S.add(t[0])
                A[t] = y_t
        
        print "Kernel Perceptron Summary"
        print "Number of support vectors for k:", len(S)
        self.A = A
        self.S = S

    def predict(self,X):
        X_full = self.X
        A = self.A
        S = self.S
        predictions = np.zeros(X.shape[0])
        
        for t in range(X.shape[0]):
            for i in S:
                predictions[t] += A[i]*np.dot(X[t],X_full[i])

        return np.sign(predictions)

# Implement this class
class BudgetKernelPerceptron(Perceptron):
    def __init__(self, beta, N, numsamples):
        self.beta = beta
        self.N = N
        self.numsamples = numsamples

        
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        assert(X.shape[0] == Y.shape[0])
    
        sample_indices = range(X.shape[0]) # for use in random sampling
        A = np.zeros(X.shape[0]) # create initial set of alpha values
        S = set() # need to be set?

        predictions = np.zeros(X.shape[0])

        for sim in range(self.numsamples):
            t = random.sample(sample_indices, 1) # pick random sample from training data and write x, y, and t
            x_t = X[t] # shape is (1,2)
            y_t = Y[t]
            a_t = A[t]

            yhat_t = 0 # initialize yhat_t to 0 to begin the sum
            for i in S: # sum over support vectors to update yhat_t
                yhat_t += A[i]*np.dot(x_t,np.transpose(X[i])) # note that i[2] is alpha_i and i[0] is x_i
                predictions[t] = yhat_t

            if y_t*yhat_t <= self.beta: # check condition. if true update set of support vectors and a_t
                S.add(t[0])
                A[t] = y_t

                if len(S) > self.N:
                    budgets = []
                    ts = []
                    for i in S:
                        budgets.append(Y[i]*(predictions[i]-A[i]*np.dot(X[i],X[i].T)))
                        ts.append(i)
                    index_to_remove = np.argmax(budgets)
                    S.remove(ts[index_to_remove]) # delete support vector w/ largest budget value
        
        print "Budget Kernel Perceptron Summary"
        print "Number of support vectors for bk:", len(S)
        self.A = A
        self.S = S
    
    def predict(self,X):
        X_full = self.X
        A = self.A
        S = self.S
        predictions = np.zeros(X.shape[0])
        
        for t in range(X.shape[0]):
            for i in S:
                predictions[t] += A[i]*np.dot(X[t],X_full[i])

        return np.sign(predictions)



# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0
N = 100
numsamples = 20000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = KernelPerceptron(numsamples)
t0 = time.clock()
k.fit(X,Y)
print "process time (s):", time.clock() - t0 
k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)

bk = BudgetKernelPerceptron(beta, N, numsamples)
t1 = time.clock()
bk.fit(X, Y)
print "process time (s):", time.clock() - t1
bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)

# Testing the accuracy
print "Accuracy calculations:"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

k.fit(X_train, Y_train)
test_predictions = k.predict(X_test)
print "k prop predicted correctly =",np.equal(test_predictions,Y_test).sum().astype('float64')/test_predictions.shape[0]

bk.fit(X_train, Y_train)
test_predictions = bk.predict(X_test)
print "bk prop predicted correctly =",np.equal(test_predictions,Y_test).sum().astype('float64')/test_predictions.shape[0]