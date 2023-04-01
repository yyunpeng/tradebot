#!/usr/bin/env python 
# -*- coding:utf-8 -*

import os
import sys
import time
import pandas as pd
import numpy as np
import scipy.linalg as linalg

class Simulate:
    def __init__(self, assetPrices):
        self.assetPrices = assetPrices
        self.dim1 = assetPrices.shape[0]
        self.dim2 = assetPrices.shape[1]
        print(self)
    def __repr__(self):
        return "'Simulate' class objectt is successfully created." 
    
    def get_sigma(self, i1, i2, delta_t=1/(252*240)):
        S = self.assetPrices[:,i1:i2+1]
        S = np.log(S)
        R = S[:,1:] - S[:,:-1]
        R_bar = np.expand_dims(np.average(R, axis=1), axis=1).reshape(self.dim1,1)
        R = R-R_bar
        sigma_head = (1/((i2 - i1)*delta_t)) * np.matmul(R, np.transpose(R))
        return sigma_head
    
    def generate_random_normals(self, tau, path_num=100):
        mean = np.array([0])
        cov = np.array([[1]])
        Z = np.random.multivariate_normal(mean, cov, size=(self.dim1 * path_num,tau))
        return np.squeeze(Z, axis=2)
    
    def simulate_one_future_date(self, St_minus_1, sigma_hat, Zt, r=0.1, path_num=100, delta_t=1/(252*240)):
        try:
            A = np.linalg.cholesky(sigma_hat)
        except:
            sigma_hat += (np.eye(self.dim1) * 0.0001)
            A = np.linalg.cholesky(sigma_hat)
        B = []
        for _ in range(path_num):
            B.append(A)
        A = linalg.block_diag(*B)

        r = np.tile(np.array([r]), reps=self.dim1*path_num)
        r = np.expand_dims(r, axis=1)

        diag_values = np.diagonal(sigma_hat).reshape(self.dim1,1)
        diag_values = np.tile(diag_values, reps=(path_num,1))

        Zt = np.expand_dims(Zt, axis=1)

        St_exp = np.exp((r-1/2*diag_values)*delta_t + np.sqrt(delta_t)*(np.matmul(A,Zt)))
        St = St_minus_1 * St_exp
        return St

    def generate_one_thousand_paths(self, St, Z_normal_vector, sigma_hat, tau, r=0.1, path_num=100, delta_t=1/(252*240)):
        St = np.tile(St, reps=path_num)
        St = np.expand_dims(St, axis=1)
        paths = [St]
        for j in range(tau):
            start  = time.perf_counter()
            St_plus_1 = self.simulate_one_future_date(St_minus_1=St, sigma_hat=sigma_hat, Zt=Z_normal_vector[:,j], r=r, path_num=path_num, delta_t=delta_t)
            paths.append(St_plus_1)
            St = St_plus_1
            end = time.perf_counter()
            # print("**********  Simulation round {} took {} seconds  ***********".format(j+1, end-start))
        paths = np.concatenate(paths, axis=1)
        return paths
    
    def execute(self, t, historical_time=30, tau_limit=30, r=0.1, path_num=100, delta_t=1/(252*240)):
        if t<historical_time:
            raise ValueError("You need to wait for half an hour before your first simulation.")
        elif t>=self.dim2:
            raise ValueError("Out of range, please select a time value <= 239.")
        else:
            tau = min(tau_limit, 239-t)
            sigma_hat = self.get_sigma(i1=t-historical_time, i2=t, delta_t=delta_t)
            St = self.assetPrices[:,t]
            Z_normal_vector = self.generate_random_normals(tau=tau, path_num=path_num)
            paths = self.generate_one_thousand_paths(St=St, Z_normal_vector=Z_normal_vector, sigma_hat=sigma_hat, tau=tau, r=r, path_num=path_num, delta_t=delta_t)
            # Generate the mean_matrix and covariance_matrix
            mean_matrix_list = []
            covariance_matrix_list = []
            for stock_index in range(self.dim1):
                stock_simulation = paths[stock_index::self.dim1,:]
                mean_vector = np.mean(stock_simulation, axis=0)
                mean_matrix_list.append(mean_vector)
                covariance_matrix = np.cov(stock_simulation.T)
                covariance_matrix_list.append(covariance_matrix)
                del stock_simulation, mean_vector, covariance_matrix
            # Returning stuff
            dim = tau+1
            return dim, mean_matrix_list, covariance_matrix_list


