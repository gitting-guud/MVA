import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import pandas as pd
import os 
from sklearn.linear_model import LinearRegression, LogisticRegression
from numpy import linalg as la


class LDA:
    def __init__(self):

        self.fitted = False
        
    def fit(self, X, Y):
        '''
        Fit the LDA model to the data 

        Input:
            X ... (in R^2) set of input points, one per column
            Y ... {0,1} the target values for the set of points X
        Output: 
        
        A fitted model with estimated parameters (π,μi,Σ)
        ''' 
        self.pi = np.mean(Y)     
        self.mu = [X[Y==i].mean(axis=0) for i in range(2)]
        self.sigma = reduce(lambda x,y:x+y, [(X[Y==i]-self.mu[i]).T @ (X[Y==i]-self.mu[i])/ (len(Y)- len(Y[Y==i])) for i in range(2)])
        self.sigma_inv = np.linalg.inv(self.sigma)
        self.coef = self.sigma_inv @ (self.mu[1][:,None] - self.mu[0][:,None])
        self.intercept = np.log(self.pi/(1-self.pi))-0.5*(self.mu[1][:,None].T@self.sigma_inv@self.mu[1][:,None])+0.5*(self.mu[0][:,None].T@self.sigma_inv@self.mu[0][:,None])
        self.fitted = True
        return self

    def predict(self, X):
        
        if self.fitted == False : 
            raise ValueError("Model not fitted yet")

        scores = np.dot(X, self.coef) + self.intercept
        probas = 1/(1+np.exp(-scores))
    
        return (np.logical_and(probas > 0.49, probas < 0.51)* 1).ravel(), ((probas > 0.5)*1).ravel()
    def plot_decision_boundary(self, X, Y, title):

        if self.fitted == False : 
            raise ValueError("Model not fitted yet")
        else : 
            x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
            y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                            np.arange(y_min, y_max, .05))

            XX = np.vstack((xx.ravel(), yy.ravel())).T
            Z = self.fit(X, Y).predict(XX)[1]
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z > 0.5, cmap=plt.cm.Paired)
            plt.axis('off')
            plt.title(title)
            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='winter')
            # plt.axis([-2, 4, -5, 3])
            plt.show()


class Logistic_Regression:
    def __init__(self):

        self.fitted = False
        
    def fit(self, X, Y, Nitermax=10,eps_conv=1e-3):
        '''
        Fit the Logistic Regression model to the data 

        Input:
            X ... (in R^2) set of input points, one per column
            Y ... {0,1} the target values for the set of points X
            Nitermax ... Number of Iterations of Newton-Raphson Algorithm
            eps_conv ... stopping criterion for the algorithm
        Output: 
        
        A fitted model with estimated parameters
        ''' 
        N_train = X.shape[0]
        X = np.hstack((np.ones((N_train,1)),X))
        #initialisation : 1 pas de l'algorithme IRLS
        w = np.zeros((X.shape[1],))
        w_old = w 
        y = 1/2*np.ones((N_train,))
        R = np.diag(y*(1-y))   # diag(y_n(1-y_n))
        z = X.dot(w_old)-la.inv(R).dot(y-Y)
        w = la.inv(X.T.dot(R).dot(X)).dot(X.T).dot(R).dot(z)

        # boucle appliquant l'algortihme de Newton-Raphson
        Niter = 1
        while ( (la.norm(w-w_old)/la.norm(w)>eps_conv) & (Niter<Nitermax) ):
            Niter = Niter+1
            y = 1/(1+np.exp(-X.dot(w)))
            R = np.diag(y*(1-y))  
            w_old = w 
            z = X.dot(w_old)-la.inv(R).dot(y-Y) 
            w = la.inv(X.T.dot(R).dot(X)).dot(X.T).dot(R).dot(z)
            
        self.w = w 
        self.Niter = Niter
        self.fitted = True
        return self

    def predict(self, X):
        '''
        Predicts labels of the data 

        Input:
            X ... (in R^2) set of input points, one per column
        Output: 
        
        Labels of test data
        ''' 
        if self.fitted == False : 
            raise ValueError("Model not fitted yet")
        N_train = X.shape[0]
        X = np.hstack((np.ones((N_train,1)),X))
        scores = np.dot(X, self.w)
        probas = 1/(1+np.exp(-scores))
        return (np.logical_and(probas > 0.49, probas < 0.51)* 1).ravel(), ((probas > 0.5)*1).ravel()




class QDA:
    def __init__(self):

        self.fitted = False
        
    def fit(self, X, Y):
        '''
        Fit the QDA model to the data 

        Input:
            X ... (in R^2) set of input points, one per column
            Y ... {0,1} the target values for the set of points X
        Output: 
        
        A fitted model with estimated parameters (π,μi,Σi)
        ''' 
        self.pi = np.mean(Y)
        self.mu = np.array([X[Y==i].mean(axis=0) for i in range(2)])
        # Σ = somme(Σi* πi)
        sigma0 = (X[Y==0] - self.mu[0]).T @ (X[Y==0] - self.mu[0]) / X.shape[0]
        sigma1 = (X[Y==1] - self.mu[1]).T @ (X[Y==1] - self.mu[1]) / X.shape[0]
        
        self.sigmas = [sigma0, sigma1]   
        self.fitted = True
        return self

    def predict(self, X):
        '''
        Fit the QDA model to the data 

        Input:
            X ... (in R^2) set of input points, one per column
            Y ... {0,1} the target values for the set of points X
        Output: 
        
        ''' 
        if self.fitted == False : 
            raise ValueError("Model not fitted yet")
        
        inverse_sigma0 = np.linalg.inv(self.sigmas[0])
        inverse_sigma1 = np.linalg.inv(self.sigmas[1])
        
        proba_class0 = np.zeros((X.shape[0],1))
        proba_class1 = np.zeros((X.shape[0],1))
        for i in range(X.shape[0]) :
            proba_class0[i,0] = (1 - self.pi) * np.exp(-0.5 * (X[i] - self.mu[0]) @ inverse_sigma0 @ (X[i] - self.mu[0]).T)
            proba_class1[i,0] = self.pi * np.exp(-0.5 * (X[i] - self.mu[1]) @ inverse_sigma1 @ (X[i] - self.mu[1]).T)

        normalisation_term = proba_class0 + proba_class1
        scores = proba_class1 / normalisation_term

        return (np.logical_and(scores > 0.49, scores < 0.51)* 1).ravel(), ((scores > 0.5)*1).ravel()
    
    def plot_decision_boundary(self, X, Y, title):

        if self.fitted == False : 
            raise ValueError("Model not fitted yet")
        else : 
            x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
            y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                            np.arange(y_min, y_max, .05))

            XX = np.vstack((xx.ravel(), yy.ravel())).T
            Z = self.fit(X, Y).predict(XX)[0]
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z > 0, cmap=plt.cm.Paired)
            plt.axis('off')
            plt.title(title)
            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='winter')
            # plt.axis([-2, 4, -5, 3])
            plt.show()