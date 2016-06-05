import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 

class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
        self.W = np.ones((3,2), dtype=float) # initial w matrix vals set to 1
        
    # create softmax function for use in gradient calc and predictions
    def softmax(self):
        vec=np.dot(self.X,self.W.T)
        num=np.exp(vec)
        denom=num.sum(axis=1)
        res=(num.T/denom).T
        return res
    
    def target_vec(self):
        T = np.zeros((self.C.shape[0],3))
        for i in range(T.shape[0]):
            T[i,self.C[i]] = 1
        self.T = T
        return

    # gradient descent algorithm
    def grad_descent(self):
        for sim in range(10000): # otherwise continue simulating for finite time
            w = np.vstack((np.zeros(3), self.W.T[1:])).T
            gradient = np.dot((self.softmax() - self.T).T, self.X) + self.lambda_parameter * w
            self.W = self.W - self.eta * gradient
            if np.absolute(self.eta * gradient).sum() < 0.00001: # run until sufficient precision is met
                break
        return
    
    # TODO: Implement this method!
    def fit(self, X, C):
        self.X = X
        self.C = C
        self.target_vec()
        self.grad_descent()
        return
        
    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization
        self.X = X_to_predict
        pred = self.softmax()
        label = pred.argmax(axis=1)
        return label
    
    
    # DON'T change this
    def visualize(self, output_file, width=2, show_charts=False):
        X = self.X

        # Create a grid of points
        x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
        y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
        xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
            y_max, .05))

        # Flatten the grid so the values match spec for self.predict
        xx_flat = xx.flatten()
        yy_flat = yy.flatten()
        X_topredict = np.vstack((xx_flat,yy_flat)).T

        # Get the class predictions
        Y_hat = self.predict(X_topredict)
        Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))
        
        cMap = c.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()