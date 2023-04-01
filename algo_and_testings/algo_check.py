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


#%%
# E_S: the mean of simulated sample stock prices S_t at time 0,1,...,9, i.e. next 10 mins
# C_i: the covariance between S_i and S_j, for all j = 0,1,..,9
# f(x): the objective function, negative expected return that we wish to minimise
    # where x is the vector denoting number of shares traded 



E_S = [11, 9, 10, 14, 13, 9, 10, 10, 10, 10] # the avereage of simulated sample S_t, at time 0,1,...,9

C_0 = [random.randrange(-4,5) for i in range(10)] # covariance list of S_0 to S_0, S_1, ..., S_9
C_1 = [random.randrange(-4,4) for i in range(10)] # covariance list of S_1 to S_0, S_1, ..., S_9
C_2 = [random.randrange(-5,5) for i in range(10)]
C_3 = [random.randrange(-5,4) for i in range(10)]
C_4 = [random.randrange(-6,6) for i in range(10)]
C_5 = [random.randrange(-6,5) for i in range(10)]
C_6 = [random.randrange(-5,6) for i in range(10)]
C_7 = [random.randrange(-4,6) for i in range(10)]
C_8 = [random.randrange(-6,4) for i in range(10)]
C_9 = [random.randrange(-7,6) for i in range(10)]


#%% objective function
def f(x):
    f = -sum(x*E_S)
    return f

x_lower = -15*np.ones(10) # setting lower bound for each component in x vector
x_upper = 15*np.ones(10)  # setting upper bound for each component in x vector

x_bound = Bounds(x_lower,x_upper)


#%% linear constraint: sum of number of shares traded should be within a limit
x_Linear_lower = -35*np.ones(1) # assume only 70 numbers of shares are available to buy/sell at present
x_Linear_upper = 35*np.ones(1) 
matrix = 1*np.ones(10)

x_Linear_bound = LinearConstraint(matrix, x_Linear_lower, x_Linear_upper) # sum of number of shares traded in the next 10 mins should be between -35 and 35


#%% non-linear constraint: portfolio skewness should be negative
x_skew_lower = -50
x_skew_upper = 0
def skew(x):
    average = ( x[0]*E_S[0] + x[1]*E_S[1] + x[2]*E_S[2] + x[3]*E_S[3] + x[4]*E_S[4] + x[5]*E_S[5] 
               + x[6]*E_S[6] + x[7]*E_S[7] + x[8]*E_S[8] + x[9]*E_S[9] ) / len(E_S) 
    
    nominator = (x[0]*E_S[0] - average)**3 + (x[1]*E_S[1] - average)**3 + (x[2]*E_S[2] - average)**3 + (x[3]*E_S[3] - average)**3 + (x[4]*E_S[4] - average)**3 + (x[5]*E_S[5] - average)**3 + (x[6]*E_S[6] - average)**3 + (x[7]*E_S[7] - average)**3 + (x[8]*E_S[8] - average)**3 + (x[9]*E_S[9] - average)**3 
    variance = (
        x[0]* (x[0]*C_0[0] + x[1]*C_0[1] + x[2]*C_0[2] + x[3]*C_0[3] + x[4]*C_0[4] + x[5]*C_0[5] + x[6]*C_0[6] + x[7]*C_0[7] + x[8]*C_0[8] + x[9]*C_0[9]) +
        x[1]* (x[0]*C_0[0] + x[1]*C_0[1] + x[2]*C_0[2] + x[3]*C_0[3] + x[4]*C_0[4] + x[5]*C_0[5] + x[6]*C_0[6] + x[7]*C_0[7] + x[8]*C_0[8] + x[9]*C_0[9]) +
        x[2]* (x[0]*C_0[0] + x[1]*C_0[1] + x[2]*C_0[2] + x[3]*C_0[3] + x[4]*C_0[4] + x[5]*C_0[5] + x[6]*C_0[6] + x[7]*C_0[7] + x[8]*C_0[8] + x[9]*C_0[9]) +
        x[3]* (x[0]*C_0[0] + x[1]*C_0[1] + x[2]*C_0[2] + x[3]*C_0[3] + x[4]*C_0[4] + x[5]*C_0[5] + x[6]*C_0[6] + x[7]*C_0[7] + x[8]*C_0[8] + x[9]*C_0[9]) +
        x[4]* (x[0]*C_0[0] + x[1]*C_0[1] + x[2]*C_0[2] + x[3]*C_0[3] + x[4]*C_0[4] + x[5]*C_0[5] + x[6]*C_0[6] + x[7]*C_0[7] + x[8]*C_0[8] + x[9]*C_0[9]) +
        x[5]* (x[0]*C_0[0] + x[1]*C_0[1] + x[2]*C_0[2] + x[3]*C_0[3] + x[4]*C_0[4] + x[5]*C_0[5] + x[6]*C_0[6] + x[7]*C_0[7] + x[8]*C_0[8] + x[9]*C_0[9]) +
        x[6]* (x[0]*C_0[0] + x[1]*C_0[1] + x[2]*C_0[2] + x[3]*C_0[3] + x[4]*C_0[4] + x[5]*C_0[5] + x[6]*C_0[6] + x[7]*C_0[7] + x[8]*C_0[8] + x[9]*C_0[9]) +
        x[7]* (x[0]*C_0[0] + x[1]*C_0[1] + x[2]*C_0[2] + x[3]*C_0[3] + x[4]*C_0[4] + x[5]*C_0[5] + x[6]*C_0[6] + x[7]*C_0[7] + x[8]*C_0[8] + x[9]*C_0[9]) +
        x[8]* (x[0]*C_0[0] + x[1]*C_0[1] + x[2]*C_0[2] + x[3]*C_0[3] + x[4]*C_0[4] + x[5]*C_0[5] + x[6]*C_0[6] + x[7]*C_0[7] + x[8]*C_0[8] + x[9]*C_0[9]) +
        x[9]* (x[0]*C_0[0] + x[1]*C_0[1] + x[2]*C_0[2] + x[3]*C_0[3] + x[4]*C_0[4] + x[5]*C_0[5] + x[6]*C_0[6] + x[7]*C_0[7] + x[8]*C_0[8] + x[9]*C_0[9]) 
        )
    
    denominator = 9*variance**3
    skewness = nominator/denominator
    
    return skewness

x_skew_bound = NonlinearConstraint(skew, x_skew_lower, x_skew_upper )


#%%
x_initialGuess = 1*np.ones(10)

result = minimize(f, x_initialGuess, method='trust-constr', 
               constraints=[x_Linear_bound, x_skew_bound],
               args=(),options={'verbose': 1}, bounds=x_bound)

x_estimate = result.x







