# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
import random as random
from Perceptron import Perceptron

# Implement this class
class KernelPerceptron(Perceptron):
    def __init__(self, numsamples):
        self.numsamples = numsamples

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        assert(X.shape[0] == Y.shape[0])
        return

    def predict(self, X):
        sample_indices = range(X.shape[0]) # for use in random sampling
        A = np.zeros(X.shape[0]) # create initial set of alpha values
        S = set() # create initial set S of support vectors
        predictions = np.empty(X.shape[0])

        for sim in range(20000):
            t = random.sample(sample_indices, 1) # pick random sample from training data and write x, y, and t
            x_t = X[t] # shape is (1,2)
            y_t = Y[t]
            a_t = A[t]

            yhat_t = 0 # initialize yhat_t to 0 to begin the sum
            for i in list(S): # sum over support vectors to update yhat_t
                yhat_t = yhat_t + A[i]*np.dot(x_t,np.transpose(X[i])) # note that i[2] is alpha_i and i[0] is x_i

            if y_t*yhat_t <= 0: # check condition. if true update set of support vectors and a_t
                S.add(t[0])
                A[t] = y_t

            predictions[t] = np.sign(yhat_t)

        return predictions

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
        return
    
    
    def predict(self,X):
        sample_indices = range(X.shape[0]) # for use in random sampling
        A = np.zeros(X.shape[0]) # create initial set of alpha values
        S = [] # need to be set?
        predictions = np.zeros(X.shape[0])

        for sim in range(self.numsamples):
            t = random.sample(sample_indices, 1) # pick random sample from training data and write x, y, and t
            x_t = X[t] # shape is (1,2)
            y_t = Y[t]
            a_t = A[t]

            yhat_t = 0 # initialize yhat_t to 0 to begin the sum
            for i in S: # sum over support vectors to update yhat_t
                yhat_t = yhat_t + i[2]*np.dot(x_t,np.transpose(i[0])) # note that i[2] is alpha_i and i[0] is x_i

            if y_t*yhat_t <= self.beta: # check condition. if true update set of support vectors and a_t

                S.append([x_t, y_t, y_t, y_t*(yhat_t-a_t*np.dot(x_t,x_t.T))])

                if len(S) > self.N:
                    del(S[np.argmax([row[3] for row in S])]) # delete support vector w/ largest value in 3rd column

            predictions[t] = np.sign(yhat_t)
        
        return predictions



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
k.fit(X,Y)
k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)

bk = BudgetKernelPerceptron(beta, N, numsamples)
bk.fit(X, Y)
bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)