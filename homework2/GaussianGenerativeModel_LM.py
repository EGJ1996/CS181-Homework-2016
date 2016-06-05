from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance

    # Just to show how to make 'private' methods
    #def __dummyPrivateMethod(self, input):
    #    return None

    # target vector, n*k
    def target(self):
        T = np.zeros((self.Y.shape[0],3))
        for i in range(T.shape[0]):
            T[i,self.Y[i]] = 1
        self.T = T
        return
        
    def N(self):
        n = self.T.sum(axis=0)
        self.n = n
        return
    
    def prior(self):
        prior = self.n/sum(self.n)
        self.prior = prior
        return
    
    def mu(self):
        phi = np.dot(self.X.T, self.T).T
        mu = (phi.T / self.n).T
        self.mu = mu
        return

    def covar(self):
        partition_indices = [self.n[0],self.n[0]+self.n[1],self.n.sum()]
        phi = [self.X[i:j] for i, j in zip([0]+partition_indices, partition_indices+[None])]
        s_k = []
        shared = np.zeros([2,2])
        for k in set(self.Y):
            shared += np.dot((phi[k]-self.mu[k]).T, (phi[k]-self.mu[k]))
            s_k.append(np.dot((phi[k]-self.mu[k]).T, (phi[k]-self.mu[k]))/phi[k].shape[0])
        self.s_k = np.array(s_k)
        self.shared = shared/self.X.shape[0]
        return


    def posterior(self,x):
        posteriors = []
        for ks in set(self.Y):
            if self.isSharedCovariance == True:
                likelihood = multivariate_normal.pdf(x, self.mu[ks], self.shared)
            if self.isSharedCovariance == False:
                likelihood = multivariate_normal.pdf(x, self.mu[ks], self.s_k[ks])
            prob = likelihood * self.prior[ks]
            posteriors.append(prob)
        return posteriors

    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.target()
        self.N()
        self.prior()
        self.mu()
        self.covar()
        return

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        self.X = X_to_predict
        all_probs = []
        for i in range(self.X.shape[0]):
            probs = self.posterior(self.X[i])
            all_probs.append(probs)
        pred = np.array(all_probs)
        pred = pred.argmax(axis=1)
        return pred

    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
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
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()


