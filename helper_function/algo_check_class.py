#!/usr/bin/env python 
# -*- coding:utf-8 -*

import os
import sys
import time
import random
import numpy as np 
import scipy
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
from scipy.optimize import OptimizeResult

class Algo_Check:
    
    def __init__(self, E_S, Sigma, dim=31, upper_bound=10, lower_bound=-10, number_of_share=10, number_of_iterations=10):
        if E_S == "" or Sigma == "":
            raise ValueError("Input ERROR!!!! please double check E_S or Sigma value")
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dim = dim
        self.sum = number_of_share
        self.x_bound = []
        self.time_slot = [0 for _ in range(self.dim)]
        self.E_S = E_S
        self.Sigma = Sigma
        self.number_of_iterations = number_of_iterations
        self.x_sum_constraints = {}
        self.x_longshort_constraints = {}
        self.x_skew_bound = {}
        
        self.setup_bound()
        self.setup_x_skew_bound()
        self.setup_x_sum_constraints()
        self.setup_x_longshort_constraints()
    
    def setup_bound(self):
        self.x_bound = []
        for i in self.time_slot:
            current_tupple = (self.lower_bound, self.upper_bound)
            self.x_bound.append(current_tupple)
        
    def setup_x_skew_bound(self):
        self.x_skew_bound = NonlinearConstraint(self.get_skewness, -10, 0)
        
    def setup_x_sum_constraints(self):
        self.x_sum_constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - self.sum}
        
    def setup_x_longshort_constraints(self):
        self.x_longshort_constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - np.sum(np.abs(x))}
        
    def setup_nominator(self, average, x):
        index = 0
        nominator = 0
        for i in self.time_slot:
            equation = ((x[index]*self.E_S[index] - average)**3)
            nominator += equation
        return nominator
        
    def get_f_variable(self, x):
        f = -np.dot(self.E_S, x) 
        return f 

    def get_skewness(self, x):
        average = np.dot(self.E_S, x) 
        nominator = self.setup_nominator(average, x)
        variance = np.dot(x, np.dot(self.Sigma, x))
        denominator = (len(self.E_S)-1)*variance**3
        skewness = nominator/denominator
        return skewness

    def downgrade_bound(self):
        self.lower_bound += 1
        self.upper_bound -= 1
        self.sum -= 1
        
    def execute_function(self):
        x_initialGuess = 1*np.ones(self.dim)
        result = minimize(self.get_f_variable, 
                          x_initialGuess, 
                          method='trust-constr', 
                          constraints=[self.x_sum_constraints, self.x_skew_bound, self.x_longshort_constraints], 
                          args=(),
                          options={'verbose': 1, 'disp': False, 'maxiter':self.number_of_iterations}, # maxitier refers to the number of iterations for minimization. Default is 1000 but will take 4-7 seconds which is too long. Restricting to 25 iterations, each iteration only takes 0.14-0.16 seconds which is much better.
                          bounds=self.x_bound)
        x_estimate = np.floor(result.x)
        # print('x_estimate: {} f(x): {}'.format(x_estimate,-result.fun))
        return x_estimate
        
    def execute(self):
        try:
            x_estimate = self.execute_function()
        except OptimizeResult as e:
            self.downgrade_bound()
            x_estimate = self.execute_function()
        return x_estimate
    