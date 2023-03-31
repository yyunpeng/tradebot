#!/usr/bin/env python 
# -*- coding:utf-8 -*

# ---------- Importing libraries and files ----------

import os
import sys
import time
import copy
import pandas as pd
import numpy as np

from helper_function.simulate import Simulate
from helper_function.algo_check_class import Algo_Check

start_main  = time.perf_counter()
start = start_main

input_path = sys.argv[1]
output_path = sys.argv[2]
symbol_file = '/opt/demos/SampleStocks.csv'

tick_data = pd.read_csv(input_path, index_col=None)
order_time = open(output_path, 'w')
symbol = pd.read_csv(symbol_file, index_col=None)['Code'].to_list()
idx_dict = dict(zip(symbol, list(range(len(symbol)))))

end = time.perf_counter()
print("The importing process took {:.2f} seconds.".format(end-start))

# ---------- Creating the tick_data pandas framework ----------

start = time.perf_counter()

def get_ms(tm):
    hhmmss = tm // 1000
    ms = (hhmmss // 10000 * 3600 + (hhmmss // 100 % 100) * 60 + hhmmss % 100) * 1000 + tm % 1000
    ms_from_open = ms - 34200000  # millisecond from stock opening
    if tm >= 130000000:
        ms_from_open -= 5400000
        ms_from_open += 1000
    return ms_from_open // 1000

def get_minute(second):
    if second < 7200:
        minute = second // 60
    elif second == 7200:
        minute = 119
    elif second < 14401:
        minute = (second - 1) // 60
    else:
        minute = 239
    return minute

tick_data = tick_data[['COLUMN01','COLUMN02','COLUMN03','COLUMN07']]
tick_data = tick_data.rename(columns={'COLUMN01': 'Index', 'COLUMN02': 'Symbol', 'COLUMN03': 'Tick_Code', 'COLUMN07': 'Price'})
tick_data['Symbol_Index'] = tick_data['Symbol'].apply(lambda x: idx_dict[x])
tick_data['Tick_Index'] = tick_data['Tick_Code'].apply(lambda x: get_ms(x))
tick_data['Minute_Index'] = tick_data['Tick_Index'].apply(lambda x: get_minute(x))
print(tick_data.head(10))
print(tick_data.tail(10))

# for minute_index in range(240):
#     tick_data_per_minute = tick_data.loc[tick_data['Minute_Index'] == minute_index]
#     tick_data_per_minute = tick_data_per_minute.drop_duplicates(subset=['Symbol_Index'])
#     print("At {} minute, there are {} number of stocks.".format(minute_index, tick_data_per_minute.shape[0]))
#     del tick_data_per_minute

end = time.perf_counter()
print("Creating tick_data pandas framework took {:.2f} seconds.".format(end-start))

# ---------- Creating assetPrices numpy array, dimension: 100 X 14402; assetPrices_perMinute numpy array, dimension: 100 X 240  ----------

start = time.perf_counter()

init_prices = [False for _ in range(100)]
count = 0
for _, row in tick_data.iterrows():
    if init_prices[row['Symbol_Index']]==False:
        init_prices[row['Symbol_Index']] = row['Price']
        count += 1
    if count == 100:
        break
# print(init_prices)

assetPrices = np.asarray(init_prices)
assetPrices = np.expand_dims(assetPrices, axis=1)
current_prices = copy.deepcopy(init_prices)
for index in range(14402):
    current_data = tick_data.loc[tick_data['Tick_Index'] == index]
    for _, row in current_data.iterrows():
        current_prices[row['Symbol_Index']] = row['Price']
    current_prices_NP = np.asarray(current_prices)
    current_prices_NP = np.expand_dims(current_prices_NP, axis=1)
    assetPrices = np.concatenate([assetPrices, current_prices_NP], axis=1)
    del current_data
assetPrices = assetPrices[:,1:]
# print(assetPrices.shape)
# print(assetPrices[:,-1])

assetPrices_perMinute_block_1 = assetPrices[:,59:7199:60]
assetPrices_perMinute_block_2 = np.expand_dims(assetPrices[:,7200], axis=1)
assetPrices_perMinute_block_3 = assetPrices[:,7260:14400:60]
assetPrices_perMinute_block_4 = np.expand_dims(assetPrices[:,14401], axis=1)
assetPrices_perMinute = np.concatenate([assetPrices_perMinute_block_1,
                                        assetPrices_perMinute_block_2,
                                        assetPrices_perMinute_block_3,
                                        assetPrices_perMinute_block_4],axis=1)
# print(assetPrices_perMinute.shape)

end = time.perf_counter()
print("Creating assetPrices and assetPrices_perMinute numpy array took {:.2f} seconds.".format(end-start))

# # ---------- Simulate ----------

# start = time.perf_counter()

# simulator = Simulate(assetPrices_perMinute)
# dim, mean_matrix_list, covariance_matrix_list = simulator.execute(t=30, r=0.1, path_num=100) # path_num = 200 is also okay

# end = time.perf_counter()
# print("The simulation process took {:.2f} seconds.".format(end-start))

# # ---------- 算法 algorithm ----------

# start = time.perf_counter()

# for stock_index in range(1):
#     algo_checker = Algo_Check(E_S=mean_matrix_list[stock_index], Sigma=covariance_matrix_list[stock_index], dim=dim, number_of_share=10)
#     x_estimate = algo_checker.execute()
#     print('Stock: {}; x_estimate: {}'.format(stock_index, x_estimate))

# end = time.perf_counter()
# print("The algorithm process took {:.2f} seconds.".format(end-start))

# ---------- Simulate + 算法 algorithm ----------


start = time.perf_counter()

simulator = Simulate(assetPrices_perMinute)

for t in range(30, 40):
    start_simulate = time.perf_counter()
    dim, mean_matrix_list, covariance_matrix_list = simulator.execute(t=30, tau_limit=30, r=0.1, path_num=100) # path_num = 200 will take 17.87 seconds, not too bad
    end_simulate = time.perf_counter()
    print("Simulation at Minute_Index = '{}' took {:.2f} seconds.".format(t, end_simulate-start_simulate))
    start_algo = time.perf_counter()
    for stock_index in range(100):
        start_algo = time.perf_counter()
        algo_checker = Algo_Check(E_S=mean_matrix_list[stock_index], 
                                  Sigma=covariance_matrix_list[stock_index], 
                                  dim=dim, 
                                  number_of_share=50, 
                                  number_of_iterations=10) # Number of iterations for minimization. Default is 1000 but will take 4-7 seconds which is too long. Restricting to 25 iterations, each iteration only takes 0.14-0.16 seconds which is much better.
        x_estimate = algo_checker.execute()
        print('Stock: {}; x_estimate: {}'.format(stock_index, x_estimate))
        
        del algo_checker
        # Do something here
        # x_estimate

    end_algo = time.perf_counter()
    print("Running algorithm at Minute_Index = '{}' took {:.2f} seconds.".format(t, end_algo-start_algo)) # Algorithm takes 33 seconds, not too bad

    del mean_matrix_list, covariance_matrix_list

end = time.perf_counter()
print("The algorithm process took {:.2f} seconds.".format(end-start))


# ---------- Writer ----------



# ---------- Post Processing ----------

end_main = time.perf_counter()
print("Overall, the entire process took {:.2f} seconds.".format(end_main-start_main))
