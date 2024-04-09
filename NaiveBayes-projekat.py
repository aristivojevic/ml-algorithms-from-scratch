# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:20:46 2024

@author: B100631
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split

#%%LEARN
def learn_smooth(data, class_att, alpha=1):
    model = {}

    apriori = data[class_att].value_counts()
    apriori = (apriori + alpha) / (apriori.sum() + len(apriori)*alpha)
    model['_apriori'] = apriori

    for attribute in data.drop(class_att, axis=1).columns:
        if data[attribute].dtype == 'object':
            mat_cont = pd.crosstab(data[attribute], data[class_att])
            mat_cont = (mat_cont + alpha) / (mat_cont.sum(axis=0) +  alpha * data[attribute].nunique())
            model[attribute] = mat_cont
        else:
            mean_std = data.groupby(class_att)[attribute].agg(['mean', 'std'])
            model[attribute] = mean_std
        

    return model


#%%PREDICT
def predict_log(model, new_instance):
    
    class_probabilities = {}
    
    for class_value in model['_apriori'].index:
        log_probability = 0

        for attribute in model:
            if attribute == '_apriori':
                log_probability += np.log(model['_apriori'][class_value])
            else:
                if new_instance[attribute] in model[attribute].index:
                    log_probability += np.log(model[attribute][class_value][new_instance[attribute]])
                else: 
                    mean = model[attribute]['mean'][class_value]
                    std = model[attribute]['std'][class_value]
                   
                    log_probability += np.log(norm.pdf(new_instance[attribute], loc=mean, scale=std))

        class_probabilities[class_value] = log_probability

    prediction = max(class_probabilities, key=class_probabilities.get)

    return prediction, class_probabilities



def predict(model, data_new):
    
    for i in range(len(data_new)):
        prediction, log_confidence = predict_log(model, data_new.iloc[i])

        confidence = {class_value: np.exp(log_prob) for class_value, log_prob in log_confidence.items()}

        data_new.loc[i, 'prediction'] = prediction
        
        for klasa in confidence:
            data_new.loc[i, 'class=' + klasa] = confidence[klasa]
            
    return data_new


#%%EXAMPLE OF USAGE
data = pd.read_csv('data/drug.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2024)

train = pd.concat([X_train, y_train], axis=1)

model = learn_smooth(train, 'Drug', alpha=10000)

predictions = predict(model, X_test.reset_index(drop=True))

result = pd.concat([y_test.reset_index(drop=True), predictions], axis=1)
accuracy = (result['Drug'] == result['prediction']).sum() / len(result)

