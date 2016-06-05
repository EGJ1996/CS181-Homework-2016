# CS 181, Spring 2016
# Homework 5: EM
# Name: Luke Mueller
# Email: lam908@mail.harvard.edu

from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.misc import logsumexp
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.misc import logsumexp

# This line loads the text for you. Don't change it! 
text_data = np.load("text.npy", allow_pickle=False)
with open('words.txt', 'r') as f:
    word_dict_lines = f.readlines()
    
# Preprocessing
text_data = text_data.astype('int')
doc_id = text_data[:,0]
word_id = text_data[:,1]
count = text_data[:,2]

D = max(doc_id)+1
V = max(word_id)+1

W = coo_matrix((count,(doc_id, word_id)), shape=(D, V)).tocsr()


class LDA(object):

    # Initializes with the number of topics
    def __init__(self, num_topics):
        self.num_topics = num_topics
        
        Beta = np.empty((num_topics,V))
        for k in range(num_topics):
            Beta[k] = np.random.dirichlet(np.ones(V))
        
        Theta = np.random.dirichlet(np.ones(num_topics))
        
        self.Beta = Beta
        self.Theta = Theta
        
    # This should run the M step of the EM algorithm
    def M_step(self):
        Gamma = self.Gamma
        # update theta_hat mat
        self.Theta = (np.sum(Gamma, axis=0)/D).T

        # update beta_hat mat
        N_d = W.sum(axis=1)
        numerator = W.transpose().dot(Gamma)
        denominator = np.dot(N_d.T, Gamma)
        self.Beta = (numerator/denominator).T
        
    # This should run the E step of the EM algorithm
    def E_step(self):
        Theta = self.Theta
        Beta = self.Beta
        
        # update gamma mat
        log_numerator = np.log(Theta.T) + W.dot(np.log(Beta.T))
        log_denominator = logsumexp(log_numerator, axis=1).reshape((-1,1))
        Gamma = np.exp(np.subtract(log_numerator, log_denominator))
        
        prod = np.multiply(Gamma, log_numerator)
        prod[np.isnan(prod)] = 0
        loglik = -logsumexp(prod)
        
        self.Gamma = Gamma
        
        return loglik
        
    def EM(self, iters):
        objective = np.empty(iters)
        for i in range(iters):
            obj = self.E_step()
            objective[i] = obj
            self.M_step()
        return objective   
    
    # This should print the topics that you find
    def print_topics(self, num_words):
        Beta = self.Beta
        top_indices = np.argsort(Beta, axis=1)[:,Beta.shape[1]-num_words::]
        for k in range(self.num_topics):
            words = []
            for w in range(num_words):
                word_index = top_indices[k,w]
                words.append(word_dict_lines[word_index])
            print "topic", k, words
            
# Feel free to add more functions as needed for the LDA class. You are welcome to change anything below this line. 
# However, your code should be contained in the constructor for the LDA class, and should be executed in a way 
# similar to the below.
R = 50
num_topics = 10
LDAClassifier = LDA(num_topics=num_topics)
loglik = LDAClassifier.EM(R)
LDAClassifier.print_topics(5)

#plt.plot(range(R-1),loglik[1:,])
#plt.xlabel("Iterations")
#plt.ylabel("Negative Expected Log Likelihood")
#plt.title("Objective Function: Topics = %d" %num_topics)
#plt.show()