#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from math import exp, log
from random import random
from sklearn.metrics import accuracy_score


def sig(x):
    return 1 / (1 + exp(-x))
    
def sig_prim(x):
    return sig(x) * (1 - sig(x))

def relu(x):
    return log(1 + exp(x))
    
def relu_prim(x):
    return 1 - exp(-relu(x))
    

class NeuralNetwork:
    """ Class representing a Multi-Layer Perceptron 
        The activation values are represented in a list of lists
        Connexions and weights are represented in a list of matrices such that :
            - Each element i of the list (a matrix) represents the incoming connexions
              of the layer i+1 from the layer i
            - the last column of each matrix represents the bias
    """
    
    def sig(self, x):
        return 1 / (1 + exp(-x))
    
    def sig_prim(self, activation):
        return (activation)*(1-activation)
    
    def relu(self, x):
        return log(1 + exp(x))
    
    def relu_prim(self, activation):
        return 1-exp(-activation)
    
    n_layers=1
    n_perceptrons=[]
    perceptrons=[]
    weights=[]
    
    def __init__(self, n_perceptrons, function):
        self.n_perceptrons=n_perceptrons
        self.n_layers=len(n_perceptrons)
        if function.lower() == "relu" :
            self.function = self.relu
            self.derivative = self.relu_prim
        else :
            self.function = self.sig
            self.derivative = self.sig_prim
        
        self.perceptrons=[]
        self.weights=[]
        
        
        ### perceptrons' initialisation ###    
        for layer in range(0, self.n_layers) :
            self.perceptrons.append(np.zeros(self.n_perceptrons[layer]).tolist())
        for layer in range(1, self.n_layers) :
             w = 0.1 * np.random.rand( self.n_perceptrons[layer], self.n_perceptrons[layer-1] +1 ) # +1 for the bias
             self.weights.append(w)
    
    
    def forward_row(self, row):
        self.perceptrons[0] = row.tolist()
        for layer in range(1, self.n_layers):
            for percept in range(0, len(self.perceptrons[layer])):
                add_bias = np.array(self.perceptrons[layer-1] + [1]) #adds the value 1 at the end of the vector
                self.perceptrons[layer][percept] = self.function(np.dot(self.weights[layer-1][percept] , add_bias))
    
    
    def forward(self, test_set):
        """
            For each sample to learn :
                For each layer L of the network (i.e. input, hidden layers and output ) :
                    For each perceptron P of the current layer L :
                        - add 1 (bias) at the end of the vector representing the activation 
                          values of the underneath layer (picked from "perceptrons" attribute of the class)
                        - extract the vector containing the current layer connexion weights (incoming
                          connexions from the underneath layer + the bias connexion)
                        - compute the scalar product of the two previous vectors and store it as the 
                          activation value of the perceptron P
            Return the activation value of the output (last element of "perceptrons" attribute)
        """
        result = []
        for row in test_set:
            self.forward_row(row)
            result.append(self.perceptrons[self.n_layers-1].copy())
        return result
    
    
    def error(self, target, activation):
        return (target-activation)*self.derivative(activation)
    
        
    def back(self, training_set, step, ind_label):
        """
            Param : 
                taining_set : a list of lists. Each list contains the values of the input and those of the target (labels)
                step : step width of the gradient descent
                ind_label : the position of the target value(s)
            This method makes the network learn from the training set.
        """
        for row in training_set:
            input_values = row.copy()

            self.perceptrons[0] = np.delete(input_values , ind_label) 
            target=list(row.copy()[i] for i in ind_label)

            self.forward_row(self.perceptrons[0]) 
            unit_values = self.perceptrons
            output=unit_values[self.n_layers-1].copy()
            error_signal=[]

            for k in range(0, len(output)):
                error_signal.append(self.error(target[k], output[k]))

            error_signal=[error_signal]
            for h_layer in range(self.n_layers-2 , 0 , -1):
                error_signal.append([])
                for j in range(0, len(self.perceptrons[h_layer])):
                    err=0
                    for k in range(0, len(self.perceptrons[h_layer+1])):
                        err = err + (error_signal[h_layer-1][k] * self.weights[h_layer][k][j])
                        err = err * self.derivative(self.perceptrons[h_layer][j])
                    error_signal[h_layer].append(err)

            for h_layer in range(1, self.n_layers-1):
                    sub_unit_values = unit_values[h_layer-1].copy() #layer just before the current one
                    sub_unit_values = np.array(sub_unit_values + [1]) #adding bias
                    delta_w = step * np.dot(np.array(error_signal[h_layer][j]) , sub_unit_values)
                    self.weights[h_layer-1] = self.weights[h_layer-1] + delta_w

            for k in range(0, len(output)):
                sub_unit_values = unit_values[self.n_layers-2].copy() #layer just before the last one
                sub_unit_values = np.array(sub_unit_values + [1]) #adding bias
                delta_w = step * np.dot(np.array(error_signal[0][k]) , sub_unit_values)
                self.weights[self.n_layers-2] = self.weights[self.n_layers-2] + delta_w                



if __name__ == "__main__":

    ###########################################################################
    ###                          "OR" OPERATOR                              ###
    ###########################################################################
    np.set_printoptions(precision=3 , threshold=5)
    
    
    print("\n============== \"Or\" Operator : =============\n")
    perceptrons=[2, 1]

    NW = NeuralNetwork(perceptrons, "sigmoid")
    
    training_list = [[0,0,0],[0,1,1],[1,0,1],[1,1,1]]
    training_set = np.array(training_list * 10)
    NW.back(training_set, step = 40 , ind_label=[2])
    test_set = np.array([[0,0],[0,1],[1,0],[1,1]])
    res = NW.forward(test_set)
    
    print("With 0 hidden layer, 40 trainint examples, step = 40 -> \n " , np.array(res))
    
    perceptrons=[2, 2, 1]

    NW = NeuralNetwork(perceptrons, "sigmoid")
    
    training_list = [[0,0,0],[0,1,1],[1,0,1],[1,1,1]]
    training_set = np.array(training_list * 100)
    NW.back(training_set, step = 10 , ind_label=[2])
    test_set = np.array([[0,0],[0,1],[1,0],[1,1]])
    res = NW.forward(test_set)
    print("\nWith 1 hidden layer 400 trainint examples, step = 10 ->  \n " ,  np.array(res))
    

    ###########################################################################
    ###                             X > Y                                   ###
    ###########################################################################
    
    
    print("\n================== X > Y : =================\n")
    
    perceptrons=[2, 1]
    
    NW = NeuralNetwork(perceptrons, "sigmoid")
    
    interval = (0, 10)
    
    #training
    training_list = []
    train_size = 1000
    for i in range(0, train_size):
        x = interval[0] + (random() * interval[1])
        y = interval[0] + (random() * interval[1])
        while y == x:
            x = interval[0] + (random() * interval[1])
            y = interval[0] + (random() * interval[1])
        if (x>y):
            training_list.append([x,y,1])
        else:
            training_list.append([x,y,0])
    
    training_set = np.array(training_list)
    NW.back(training_set, step = 1, ind_label=[2])
    
    #test
    test_list = []
    true=[]
    test_size = 20000
    for i in range(0, test_size):
        x = interval[0] + (random() * interval[1])
        y = interval[0] + (random() * interval[1])
        while y == x:
            x = interval[0] + (random() * interval[1])
            y = interval[0] + (random() * interval[1])
        if (x>y):
            test_list.append([x,y])
            true.append(1)
        else:
            test_list.append([x,y])
            true.append(0)
    res = NW.forward(np.array(test_list))
    
    #accuracy
    threshold = (0.1 , 0.9)
    pred = []
    for i in res:
        if i[0] > threshold[0] and i[0] < threshold[1] :
            pred.append(-1)
        elif i[0] > 0.5:
            pred.append(1)
        else:
            pred.append(0)
    
    print("-> Accuracy (random) = " , accuracy_score(true, pred, normalize=True, sample_weight=None))
    
    
    ###### test when x and y are close
    
    test_list = []
    test_size = 20000
    true=[]
    for i in range(0, train_size):
        x = interval[0] + (random() * interval[1])
        y = interval[0] + (random() * interval[1])
        test_list.append([x,x+random()])
        true.append(1)
        test_list.append([y-random(),y])
        true.append(0)
    
    res = NW.forward(np.array(test_list))
    
    #accuracy
    threshold = (0.1 , 0.9)
    pred = []
    for i in res:
        if i[0] > threshold[0] and i[0] < threshold[1] :
            pred.append(-1)
        elif i[0] > 0.5:
            pred.append(1)
        else:
            pred.append(0)
    
    print("-> Accuracy (x and y close) = " , accuracy_score(true, pred, normalize=True, sample_weight=None))
    
    
    
    
    
    
    
