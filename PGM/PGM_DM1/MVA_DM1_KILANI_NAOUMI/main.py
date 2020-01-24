from __future__ import absolute_import
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import pandas as pd
import os 
from sklearn.linear_model import LinearRegression
from utils import LDA, Logistic_Regression 


path = r'D:/MVA/PGM/homework1_data/data'

train_files =['trainA', 'trainB', 'trainC']
test_files =['testA', 'testB', 'testC']


if __name__=='__main__':

    algorithm = 'QDA'
    

    if algorithm == 'LDA':
        lda = LDA()
        for i, (file_, test) in enumerate(zip(train_files, test_files)): 
            reader_train = pd.read_csv(os.path.join(path, file_), delimiter=' ', header=None)
            reader_test= pd.read_csv(os.path.join(path, test), delimiter=' ', header=None)
            values_train = reader_train.values
            values_test = reader_test.values
            X, Y = values_train[:,:2] , values_train[:,-1]
            lda = lda.fit(X, Y)
            error_train = np.abs(lda.predict(X)[1] - Y).mean() 
            accuracy_train = 1 - error_train
            error_test = np.abs(lda.predict(values_test[:,:2])[1] -  values_test[:,-1]).mean()
            print(' for Dataset {} Train error : {}%, Test error: {}%'.format(i , error_train*100,error_test*100))
            with plt.style.context('seaborn') :
                lda.plot_decision_boundary(X, Y, title='Decision function for {}, accuracy = {:.3f}'.format(file_, accuracy_train))
    
    if algorithm == 'LinearReg':    
        lr = LinearRegression()
        for i, (file_, test) in enumerate(zip(train_files, test_files)): 
            reader_train = pd.read_csv(os.path.join(path, file_), delimiter=' ', header=None)
            reader_test= pd.read_csv(os.path.join(path, test), delimiter=' ', header=None)
            values_train = reader_train.values
            values_test = reader_test.values
            X, Y = values_train[:,:2] , values_train[:,-1]
            Y_ = Y.copy()
            Y_[Y_==0] = -1
            lr = lr.fit(X, Y_)
            # print(lr.coef_, lr.intercept_)
            error_train = np.abs((lr.predict(X)>0) - Y).mean()
            accuracy = 1 - error_train

            error_test = np.abs((lr.predict(values_test[:,:2])>0) -  values_test[:,-1]).mean()
            print(' for Dataset {} Train error : {}%, Test error: {}%'.format(i , error_train*100,error_test*100))
            x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
            y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                            np.arange(y_min, y_max, .05))

            XX = np.vstack((xx.ravel(), yy.ravel())).T
            Z = (lr.predict(XX)>0) * 1
            Z = Z.reshape(xx.shape)
            with plt.style.context('seaborn') :
                plt.contourf(xx, yy, Z > 0, cmap=plt.cm.Paired)
                plt.axis('off')
                plt.title('Decision function for {}, accuracy = {:.3f}'.format(file_, accuracy))
                plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='winter')
                plt.show()

    if algorithm == 'LogReg': 
        logr = Logistic_Regression()
        for i, (file_, test) in enumerate(zip(train_files, test_files)): 
            reader_train = pd.read_csv(os.path.join(path, file_), delimiter=' ', header=None)
            reader_test= pd.read_csv(os.path.join(path, test), delimiter=' ', header=None)
            values_train = reader_train.values
            values_test = reader_test.values
            X, Y = values_train[:,:2] , values_train[:,-1]
            logr = logr.fit(X, Y, Nitermax=10)
            print(logr.w)
            error_train = np.abs(logr.predict(X)[1] - Y).mean()
            accuracy = 1 - error_train
            error_test = np.abs(logr.predict(values_test[:,:2])[1] -  values_test[:,-1]).mean()
            print(' for Dataset {} Train error : {}%, Test error: {}%'.format(i , error_train*100,error_test*100))
            x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
            y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                            np.arange(y_min, y_max, .05))

            XX = np.vstack((xx.ravel(), yy.ravel())).T
            Z = logr.predict(XX)[1]
            Z = Z.reshape(xx.shape)
            with plt.style.context('seaborn') :
                plt.contourf(xx, yy, Z > 0, cmap=plt.cm.Paired)
                plt.axis('off')
                plt.title('Decision function for {}, accuracy = {:.3f}'.format(file_, accuracy))
                plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='winter')
                plt.show()

    
    if algorithm == 'QDA': 
        qda = QDA()
        for i, (file_, test) in enumerate(zip(train_files, test_files)): 
            reader_train = pd.read_csv(os.path.join(path, file_), delimiter=' ', header=None)
            reader_test= pd.read_csv(os.path.join(path, test), delimiter=' ', header=None)
            values_train = reader_train.values
            values_test = reader_test.values
            X, Y = values_train[:,:2] , values_train[:,-1]
            qda = qda.fit(X, Y)
            print(qda.sigmas, qda.mu, qda.pi)
            error_train = np.abs(qda.predict(X)[1] - Y).mean()
            accuracy = 1 - error_train
            error_test = np.abs(qda.predict(values_test[:,:2])[1] -  values_test[:,-1]).mean()
            print(' for Dataset {} Train error : {}%, Test error: {}%'.format(i , error_train*100,error_test*100))
            x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
            y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, .05),
                            np.arange(y_min, y_max, .05))

            XX = np.vstack((xx.ravel(), yy.ravel())).T
            Z = qda.predict(XX)[1]
            Z = Z.reshape(xx.shape)
            with plt.style.context('seaborn') :
                plt.contourf(xx, yy, Z > 0, cmap=plt.cm.Paired)
                plt.axis('off')
                plt.title('Decision function for {}, accuracy = {:.3f}'.format(file_, accuracy))
                plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='winter')
                plt.show()
    