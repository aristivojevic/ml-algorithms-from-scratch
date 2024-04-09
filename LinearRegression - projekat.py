# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 11:31:38 2024

@author: B100631
"""

import pandas as pd
import numpy as np


class LinearRegression:
    def __init__(self, alpha=0.6, max_iters=10000, tolerance=0.01, regularization_param=0):
        self.alpha = alpha
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.w = None
        self.mean = None
        self.std = None
        self.regularization_param=regularization_param

    def learn(self, X_train, y_train):
        self.mean = X_train.mean()
        self.std = X_train.std()
        
        X_train_normalized = (X_train - self.mean) / self.std
        X_train_normalized['X0'] = 1  # Dodajemo kolonu sa jedinicama
        
        m, n = X_train_normalized.shape
        X_train_normalized = X_train_normalized.to_numpy()
        y_train=y_train.to_numpy()

        self.w = np.random.random((1, n))
        
        for it in range(self.max_iters):
            
            pred = X_train_normalized.dot(self.w.T)
            err = pred - y_train
            w_penalty = self.w.copy()
            w_penalty[:, (self.w.shape[1]-1)] = 0
            grad = err.T.dot(X_train_normalized) / m + self.regularization_param * w_penalty
            self.w -= self.alpha * grad
            
            grad_norm = np.abs(grad).sum()
            MSE = err.T.dot(err) / m
            print(it, MSE, grad_norm)
            if grad_norm < self.tolerance:
                break
            
    def learn_online(self, X_train, y_train):
        self.mean = X_train.mean()
        self.std = X_train.std()
        
        X_train_normalized = (X_train - self.mean) / self.std
        X_train_normalized['X0'] = 1  # Dodajemo kolonu sa jedinicama
        
        m, n = X_train_normalized.shape
        #X_train_normalized = X_train_normalized.to_numpy()

        self.w = np.random.random((1, n))
        
        X_shuffled = X_train_normalized.sample(frac=1, random_state=10)
        i = -1
    
        
        for it in range(self.max_iters):
            
            if i >= (X_train_normalized.shape[0] - 1):
                i = 0
            else:
                i = i + 1;
                
            random_instance = X_shuffled.iloc[i]
            index = random_instance.name
            
            random_instance = random_instance.to_numpy()
            random_instance = np.reshape(random_instance, (random_instance.shape[0],1))
            
            pred = random_instance.T.dot(self.w.T)
            err = pred - y_train.loc[index].to_numpy()
            err = np.reshape(err, (1,1))
            
            w_penalty = self.w.copy()
            w_penalty[:, (self.w.shape[1]-1)] = 0
            
            grad = err.dot(random_instance.T) / m + self.regularization_param * w_penalty
            self.w = self.w - self.alpha*grad
        	
            MSE = err.T.dot(err) / m
            grad_norm = np.abs(grad).sum()
            print(it, MSE, grad_norm)
        
            if grad_norm < self.tolerance:
                break     
                        
    def predict(self, X_test):
        m, _ = X_test.shape
        X_test_normalized = (X_test - self.mean) / self.std
        X_test_normalized['X0'] = 1
        X_test_normalized = X_test_normalized.to_numpy()
        return X_test_normalized.dot(self.w.T)


# Example usage:
data = pd.read_csv('house.csv')
y_train = data[['Price']]
X_train= data.drop('Price', axis=1)

model = LinearRegression()
model.learn_online(X_train, y_train)

data_new = pd.read_csv('house_new.csv')
predictions = model.predict(data_new)
print(predictions)

