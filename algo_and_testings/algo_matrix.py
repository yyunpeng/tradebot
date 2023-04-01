# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 15:44:47 2023

@author: xuyun
"""

#%%
import numpy as np 
import random
import sympy
from mpmath import *


#%%

E_S = [11, 9, 10, 14, 13, 9, 10, 10, 10, 10] # the avereage of simulated sample S_t, at time 0,1,...,9

c_0 = [2*random.randrange(-4,5) for i in range(10)] 
c_1 = [2*random.randrange(-4,4) for i in range(10)] 
c_2 = [2*random.randrange(-5,5) for i in range(10)]
c_3 = [2*random.randrange(-5,4) for i in range(10)]
c_4 = [2*random.randrange(-6,6) for i in range(10)]
c_5 = [2*random.randrange(-6,5) for i in range(10)]
c_6 = [2*random.randrange(-5,6) for i in range(10)]
c_7 = [2*random.randrange(-4,6) for i in range(10)]
c_8 = [2*random.randrange(-6,4) for i in range(10)]
c_9 = [2*random.randrange(-7,6) for i in range(10)]

curr_limit = 100-20

# %pprint
W=[
   c_0 + [-1*E_S[0]] + [-1, 0],
   c_1 + [-1*E_S[1]] + [-1, 0],
   c_2 + [-1*E_S[2]] + [-1, 0],
   c_3 + [-1*E_S[3]] + [-1, 0],
   c_4 + [-1*E_S[4]] + [-1, 0],
   c_5 + [-1*E_S[5]] + [-1, 0],
   c_6 + [-1*E_S[6]] + [-1, 0],
   c_7 + [-1*E_S[7]] + [-1, 0],
   c_8 + [-1*E_S[8]] + [-1, 0],
   c_9 + [-1*E_S[9]] + [-1, 0],
   E_S + [0,0,max(E_S)],
   [1]*len(E_S)+[0,0,curr_limit]
   ]
W_matrix = sympy.Matrix(W)

W_rref = W_matrix.rref()
x_estimate = W_rref[0].col(-1) # first 10 numbers


