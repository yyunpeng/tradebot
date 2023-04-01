# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:56:32 2023

@author: xuyun
"""
#%%
import numpy as np 
import scipy
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
import random
from algo_check_class import Algo_Check


#%%
# E_S: the mean of simulated sample stock prices S_t at time 0,1,...,9, i.e. next 10 mins
# C_i: the covariance between S_i and S_j, for all j = 0,1,..,9
# f(x): the objective function, negative expected return that we wish to minimise
    # where x is the vector denoting number of shares traded 


E_S = np.array([11, 9, 10, 14, 13, 9, 10, 10, 10, 10]) # the avereage of simulated sample S_t, at time 0,1,...,9

Sigma = np.array([
                  [random.randrange(-4,5) for i in range(10)], [random.randrange(-4,4) for i in range(10)], [random.randrange(-5,5) for i in range(10)],
                  [random.randrange(-5,4) for i in range(10)], [random.randrange(-6,6) for i in range(10)], [random.randrange(-6,5) for i in range(10)], 
                  [random.randrange(-5,6) for i in range(10)], [random.randrange(-4,6) for i in range(10)], [random.randrange(-6,4) for i in range(10)], 
                  [random.randrange(-7,6) for i in range(10)]
                ]) # covariance matrix of prices



#%% objective function

def f(x):
    f = -np.dot(E_S, x) 
    return f

def skew(x):
    average = np.dot(E_S, x) 
    nominator = (x[0]*E_S[0] - average)**3 + (x[1]*E_S[1] - average)**3 + (x[2]*E_S[2] - average)**3 + (x[3]*E_S[3] - average)**3 + (x[4]*E_S[4] - average)**3 + (x[5]*E_S[5] - average)**3 + (x[6]*E_S[6] - average)**3 + (x[7]*E_S[7] - average)**3 + (x[8]*E_S[8] - average)**3 + (x[9]*E_S[9] - average)**3 
    variance = np.dot(x, np.dot(Sigma, x))
    denominator = (len(E_S)-1)*variance**3
    skewness = nominator/denominator
    return skewness

x_bound = bounds = [(-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20), (-20, 20)]
x_sum_constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 20} # sum of number of shares traded constraint
x_longshort_constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - np.sum(np.abs(x))} # long buy and short sell constraint
x_skew_bound = NonlinearConstraint(skew, -10, 0)


#%%
x_initialGuess = 1*np.ones(10)

result = minimize(f, x_initialGuess, method='trust-constr', 
               constraints=[x_sum_constraints, x_skew_bound, x_longshort_constraints],
               args=(),options={'verbose': 1}, bounds=x_bound)

#%%
x_estimate = np.floor(result.x)

print('x_estimate;',x_estimate,'f(x):', -result.fun)

if __name__ == "__main__":
    algo = Algo_Check(E_S)
    

























