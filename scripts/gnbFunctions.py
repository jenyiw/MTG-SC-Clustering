# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:28:17 2023

@author: Sumitra
"""

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Number of cells = N
# Number of features = d
# Number of classes = k

def accuracy_score(y_true, y_pred):
    '''
    Get the accuracy score for the classification
    Input: Predicted labels and True labels
    Output: Accuracy score
    '''
    return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100 ,2)

def calculate_class_priors(target):
    '''
    Get the class priors for each class; calculate probability of each class
    Input: List of training labels [N,]
    Output: Probability of each class [k,]
    '''
    classes = np.unique(target)
    priors = np.empty(classes.shape)
    # Iterate over all unique classes and sum instances of each
    for target_class in classes:
        class_count = sum(target == target_class)
        priors[int(target_class)] = class_count/len(target)

    return priors

def calculate_param(exp, target):
    '''
    Calculate mean and variance for distribution of each class each class
    Input: Count matrix [N, d], labels [N,]
    Output: Matrix containing the mean and variance for each feature, for each class [d, k, 2]
    '''
    likelihoods = np.zeros((exp.shape[1], len(np.unique(target)), 2))
    for feature in range(exp.shape[1]):
        for target_class in np.unique(target):
            # Calculate the Mean for the feature values belonging to the target_class
            likelihoods[feature][target_class][0] = exp[np.where(target == target_class)[0], feature].mean()
            # Calculate the variance
            likelihoods[feature][target_class][1] = exp[np.where(target == target_class)[0], feature].var()

    return likelihoods


def predict(exp_test, features, classes, priors, likelihood):
    '''
    Calculate posterior probability for each sample, belonging to each class and predict the most probable class
    Input: Test count matrix [N, d], Encoded features list [d,], Encoded class list [k,], Priors for each class [k,], Parameter Matrix [d, k, 2]
    Output: 1-D array of predicted classes for each sample [N,]
    '''
    # Create an array to store the predicted classes
    predicted_classes = []
    
    
    for datapoint in exp_test:
        prob_class = np.zeros((len(classes)))
        
        for target_class in classes:
            prior = priors[target_class]
            log_likelihood_tot = 0

            for feat, feat_val in zip(features, datapoint):
                var = likelihood[feat][target_class][1]
                mean = likelihood[feat][target_class][0]
                temp_log = np.log((1/math.sqrt(2*math.pi*var))) + (-(feat_val - mean)**2 / (2*var))
                log_likelihood_tot += temp_log

                prob_class[target_class] = log_likelihood_tot + np.log(prior)

        predicted_classes.append(np.argmax(prob_class))
    return predicted_classes




if __name__ == 'main':
   X = np.loadtxt('reduced_sample_20_PCs.csv', delimiter=',')
   y = np.loadtxt('labels.txt', delimiter=' ')
   features = [x for x in range(X.shape[1])]
   classes = np.unique(y)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0, stratify=y)

   priors_data = calculate_class_priors(y_train)
   param = calculate_param(X_train, y_train)
   pred_pca = np.array(predict(X_test, features, classes, priors_data, param))
   print(accuracy_score(pred_pca, y_test))
    