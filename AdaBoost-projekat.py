# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:43:35 2024

@author: B100631
"""

import pandas as pd
import numpy as np
import math
import random
import sys
pd.set_option('display.max_columns', None)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


class AdaBoostEnsemble:
    def __init__(self, ensemble_size=5, base_algorithms=None, diversity_factor=1):
        
        if not 0 <= diversity_factor <= 1:
            print("Diversity factor must be between 0 and 1.")
            sys.exit()
            
        self.ensemble_size = ensemble_size
        self.base_algorithms = base_algorithms
        self.diversity_factor = diversity_factor
        self.ensemble = []
        self.weights = np.zeros(ensemble_size)
        self.allowed_algorithms = [DecisionTreeClassifier, GaussianNB, SVC, LogisticRegression]

    def preprocess_data(self, file_path, class_value):
        data = pd.read_csv(file_path)
        X = data.drop(class_value, axis=1)
        
        target_values = data[class_value].unique()
        if not (len(target_values) == 2 and all(value in [0, 1] for value in target_values)):
            print(f"The target variable '{class_value}' does not contain only values 0 and 1")
            sys.exit()

        y = data[class_value]
        y = data[class_value]*2-1
        X = pd.get_dummies(X)
        print("Target variable will have values -1, 1 instead of 0, 1.")
        return X, y
    
    def validate_algorithms(self):
        if self.base_algorithms is not None:
            for algorithm in self.base_algorithms:
                if algorithm not in self.allowed_algorithms:
                    return False
        else:
            self.base_algorithms=self.allowed_algorithms
            
        return True

    def train(self, X, y):
        n, m = X.shape
        alfas = pd.Series(np.array([1/n]*n), index=X.index)
        
        if not self.validate_algorithms():
            print("Training aborted due to invalid algorithms.")
            print(f"Allowed algorithms: {self.allowed_algorithms}")
            sys.exit()
               
        for t in range(self.ensemble_size):
            base_algorithm = random.choice(self.base_algorithms)
            
            if base_algorithm == DecisionTreeClassifier:
                week_learner = base_algorithm(max_depth=2)
            else:
                week_learner = base_algorithm()
    
            print('Model used', t+1, week_learner)
            
            model = week_learner.fit(X, y, sample_weight=alfas)
            predictions = model.predict(X)
            
            error = (predictions != y).astype(int)
           
            weighted_error = (error * alfas).sum()
                   
            w = 1/2 * math.log((1-weighted_error)/weighted_error)
            
            self.ensemble.append(model)
            self.weights[t] = w
            factor = np.exp(-self.diversity_factor*w*predictions*y)
            alfas = alfas * factor
            alfas = alfas / alfas.sum()
            
        print("Weights of models:", self.weights)
    
    def predict(self, X):
        
        if not self.ensemble:
            sys.exit()
            
        predictions_df = pd.DataFrame()
        
        for i, model in enumerate(self.ensemble):
            predictions_df[f'Model {i+1}'] = model.predict(X)
            
        confidence_class_1=predictions_df.replace(-1,0).dot(self.weights)/sum(self.weights)
        confidence_class_minus_1=abs(predictions_df.replace(1,0)).dot(self.weights)/sum(self.weights)
        
        ensemble_predictions = predictions_df.dot(self.weights)

        predictions_df['Ensemble'] = np.sign(ensemble_predictions)
        predictions_df['conf_class_1'] = confidence_class_1
        predictions_df['conf_class_-1'] = confidence_class_minus_1
    
        return predictions_df


    def evaluate_accuracy(self, predictions, y):
        
        predictions=predictions.drop(columns=['conf_class_1', 'conf_class_-1'])
        y=y.reset_index(drop=True)
        
        for i, model in enumerate(self.ensemble):
            correct_predictions_each_model= (predictions[f'Model {i+1}'] == y).astype(int)
            print(f'Model accuracy {i+1}',correct_predictions_each_model.mean())

        correct_predictions = (predictions['Ensemble'] == y).astype(int)
        print('Overall ensemble accuracy',correct_predictions.mean())
        
        
#%% EXAMPLE OF USE 

ada_boost = AdaBoostEnsemble(ensemble_size=5, 
                             base_algorithms=[DecisionTreeClassifier, GaussianNB, SVC, LogisticRegression],
                             diversity_factor=1)

X, y = ada_boost.preprocess_data('data/drugY.csv', 'Drug')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2024)

ada_boost.train(X_train, y_train)

predictions = ada_boost.predict(X_test)

ada_boost.evaluate_accuracy(predictions, y_test)
combined_df=pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True), predictions], axis=1)

print(combined_df)
