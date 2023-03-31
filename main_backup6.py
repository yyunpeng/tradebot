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

print("This entire code file is going to take approximately 1 hour to run.\nWe thank you for your patience.")

start_main  = time.perf_counter()
start = start_main

input_path = sys.argv[1]
output_path = sys.argv[2]
symbol_file = '/opt/demos/SampleStocks.csv'

tick_data = pd.read_csv(input_path, index_col=None)
# order_time = open(output_path, 'w')
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
tick_data['buyable'] = [0] * len(tick_data)
tick_data['sellable'] = [0] * len(tick_data)
tick_data['trade_volume'] = [0] * len(tick_data)

# print(tick_data.head(10))
# print(tick_data.tail(10))

end = time.perf_counter()
print("Creating tick_data pandas framework took {:.2f} seconds.".format(end-start))

# ---------- Creating assetPrices numpy array, dimension: 100 X 14402; assetPrices_perMinute numpy array, dimension: 100 X 240  ----------

print("Creating asset prices per minute interval NumPy array.\nPlease wait, it will take approximately 30 seconds...")

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

# ---------- Simulate + 算法 algorithm ----------

start = time.perf_counter()

simulator = Simulate(assetPrices_perMinute)

sell_arr = np.empty((100, 240))
sell_arr[0:100, 0:240] = 0.
buy_arr = np.empty((100, 240))
buy_arr[0:100, 0:240] = 0.

for t in range(30, 239, 2):
    start_simulate = time.perf_counter()
    print("t = {}. Start simulating stock movement paths.".format(t))
    dim, mean_matrix_list, covariance_matrix_list = simulator.execute(t=t, 
                                                                      historical_time=30, # How long is the history taken to generate the sample covariance matrix
                                                                      tau_limit=10, # How many number of future minutes you will simulate until
                                                                      r=0.10, # China interest rate
                                                                      path_num=250) # path_num = 250 will take 15-20 seconds, not too bad
    end_simulate = time.perf_counter()
    print("Simulation at Minute_Index = '{}' took {:.2f} seconds.".format(t, end_simulate-start_simulate))
    start_algo = time.perf_counter()
    for stock_index in range(100):
        start_minimise = time.perf_counter()
        algo_checker = Algo_Check(E_S=mean_matrix_list[stock_index], 
                                  Sigma=covariance_matrix_list[stock_index], 
                                  dim=dim, 
                                  upper_bound=50, 
                                  lower_bound=-50, 
                                  number_of_share=50, 
                                  number_of_iterations=30) # Number of iterations for minimization. Default is 1000 but will take 4-7 seconds which is too long. Restricting to 25 iterations, each iteration only takes 0.14-0.16 seconds which is much better.
        x_estimate = algo_checker.execute()
        print("t={}. stock_index = {}. x_estimate = {}".format(t, stock_index, x_estimate))
        if x_estimate[0] < 0.:
            sell_arr[stock_index,t] = abs(x_estimate[0])
        elif x_estimate[0] >= 0.:
            buy_arr[stock_index,t] = abs(x_estimate[0])
        del algo_checker
        end_minimise = time.perf_counter()
        # print("Running minimisation for stock {} at Minute_Index = '{}' took {:.2f} seconds.".format(stock_index, t, end_minimise-start_minimise)) # Algorithm takes 33 seconds, not too bad
    end_algo = time.perf_counter()
    print("Simulation at Minute_Index = '{}' took {:.2f} seconds.".format(t, end_simulate-start_simulate))
    print("Running minimisation algorithm at Minute_Index = '{}' took {:.2f} seconds.".format(t, end_algo-start_algo)) # Algorithm takes 33 seconds, not too bad
    del mean_matrix_list, covariance_matrix_list

print(sell_arr)
print(buy_arr)

print("Working in progress...")

for i in range(100):
    # Ignore first 30 entries for normalization
    extracted_array_original = sell_arr[i,30:]
    row_sum = np.sum(extracted_array_original)
    if row_sum == 0:
        for index_inside in range(97):
            extracted_array_original[index_inside*2] = 1.
        row_sum = np.sum(extracted_array_original)
    normalization_factor = 97 / row_sum
    extracted_array_normalised = normalization_factor * extracted_array_original
    extracted_array_integer = np.floor(extracted_array_normalised).astype(int)
    difference = 97 - int(np.sum(extracted_array_integer))
    extracted_array_after_decimal = extracted_array_normalised - extracted_array_integer
    # print(difference)
    # print(extracted_array_integer)
    # print(extracted_array_after_decimal)
    if difference>0:
        index_list = np.argpartition(extracted_array_after_decimal, -difference)[-difference:]
        for j in range(extracted_array_integer.shape[0]):
            if j in index_list:
                extracted_array_integer[j] +=1
        # print(np.nansum(extracted_array_integer))
    sell_arr[i,30:] = extracted_array_integer 
sell_arr = np.floor(sell_arr).astype(int)

for i in range(100):
    # Ignore first 30 entries for normalization
    extracted_array_original = buy_arr[i,30:]
    row_sum = np.sum(extracted_array_original)
    if row_sum == 0:
        for index_inside in range(97):
            extracted_array_original[index_inside*2] = 1.
        row_sum = np.sum(extracted_array_original)
    normalization_factor = 97 / row_sum
    extracted_array_normalised = normalization_factor * extracted_array_original
    extracted_array_integer = np.floor(extracted_array_normalised).astype(int)
    difference = 97 - int(np.sum(extracted_array_integer))
    extracted_array_after_decimal = extracted_array_normalised - extracted_array_integer
    # print(difference)
    # print(extracted_array_integer)
    # print(extracted_array_after_decimal)
    if difference>0:
        index_list = np.argpartition(extracted_array_after_decimal, -difference)[-difference:]
        for j in range(extracted_array_integer.shape[0]):
            if j in index_list:
                extracted_array_integer[j] +=1
        # print(np.nansum(extracted_array_integer))
    buy_arr[i,30:] = extracted_array_integer 
buy_arr = np.floor(buy_arr).astype(int)

print("Working in progress.........")

sell_tick_list = []
for i in range(100):
    each_stock = []
    for j in range(240):
        each_stock.append([])
    sell_tick_list.append(each_stock)

for stock_index in range(100): 
    for minute_index in range(240): 
        if sell_arr[stock_index][minute_index]== 0:
            pass
        else: 
            tick_data_per_minute = tick_data[(tick_data['Minute_Index'] == minute_index) & (tick_data['Symbol_Index'] == stock_index)]
            for _ , row in tick_data_per_minute.iterrows():
                tup_index = row['Index']
                tup_price = row['Price']
                tup_tick_idx = row['Tick_Index']
                tup_volume = sell_arr[stock_index,minute_index]
                tup = (tup_index,tup_tick_idx,tup_price,tup_volume)
                sell_tick_list[stock_index][minute_index].append(tup)

buy_tick_list = []
for i in range(100):
    each_stock = []
    for j in range(240):
        each_stock.append([])
    buy_tick_list.append(each_stock)

for stock_index in range(100): 
    for minute_index in range(240): 
        if buy_arr[stock_index][minute_index]== 0:
            pass
        else: 
            tick_data_per_minute = tick_data[(tick_data['Minute_Index'] == minute_index) & (tick_data['Symbol_Index'] == stock_index)]
            for _ , row in tick_data_per_minute.iterrows():
                tup_index = row['Index']
                tup_price = row['Price']
                tup_tick_idx = row['Tick_Index']
                tup_volume = buy_arr[stock_index,minute_index]
                tup = (tup_index,tup_tick_idx,tup_price,tup_volume)
                buy_tick_list[stock_index][minute_index].append(tup)

# print(sell_tick_list[50][2])
# print(sell_tick_list[99][30])
# print(sell_tick_list[99][31])
# print(sell_tick_list[99][32])
# print(sell_tick_list[99][33])
# print(sell_tick_list[99][34])
# print(sell_tick_list[99][35])
# print(sell_tick_list[99][36])
# print(sell_tick_list[99][37])
# print(sell_tick_list[99][38])

print("Working in progress..................")

for stock_index in range(0,100):
    for minute_index in range(240): 
        if sell_tick_list[stock_index][minute_index]==[]:
            pass
        else:
            # print(sell_tick_list[stock_index][i])
            max_second = -1
            max_tuple = ()
            for tup in sell_tick_list[stock_index][minute_index]:
                if tup[1] > max_second:
                    # print(tup)
                    max_second = tup[1]
                    max_tuple = tup
            tick_data.iloc[max_tuple[0],8] = 1
            tick_data.iloc[max_tuple[0],9] = max_tuple[3]

for stock_index in range(0,100):
    for minute_index in range(240): 
        if buy_tick_list[stock_index][minute_index]==[]:
            pass
        else:
            max_second = -1
            max_tuple = ()
            for tup in buy_tick_list[stock_index][minute_index]:
                if tup[1] > max_second and tick_data.iloc[tup[0],8]!=1:
                    # print(tup)
                    max_second = tup[1]
                    max_tuple = tup
            tick_data.iloc[max_tuple[0],7] = 1
            tick_data.iloc[max_tuple[0],9] = max_tuple[3]

print("Almost done..................")

for stock_index in range(0,100):
    tick_data_1 = tick_data[((tick_data['Minute_Index'] == 0) | 
                             (tick_data['Minute_Index'] == 1) | 
                             (tick_data['Minute_Index'] == 2) |
                             (tick_data['Minute_Index'] == 3) |
                             (tick_data['Minute_Index'] == 4) |
                             (tick_data['Minute_Index'] == 5) |
                             (tick_data['Minute_Index'] == 6) |
                             (tick_data['Minute_Index'] == 7) |
                             (tick_data['Minute_Index'] == 8) ) & (tick_data['Symbol_Index'] == stock_index)]
    count = 1
    for _ , row in tick_data_1.iterrows():
        if count==1:
            row_index = row['Index']
            tick_data.iloc[row_index,7] = 1
            tick_data.iloc[row_index,9] = 1
        elif count==2:
            row_index = row['Index']
            tick_data.iloc[row_index,8] = 1
            tick_data.iloc[row_index,9] = 1
        else:
            break
        count += 1
    tick_data_2 = tick_data[((tick_data['Minute_Index'] == 10) |
                             (tick_data['Minute_Index'] == 11) |
                             (tick_data['Minute_Index'] == 12) |
                             (tick_data['Minute_Index'] == 13) |
                             (tick_data['Minute_Index'] == 14) |
                             (tick_data['Minute_Index'] == 15) |
                             (tick_data['Minute_Index'] == 16) |
                             (tick_data['Minute_Index'] == 17) |
                             (tick_data['Minute_Index'] == 18) ) & (tick_data['Symbol_Index'] == stock_index)]
    count = 1
    for _ , row in tick_data_2.iterrows():
        if count==1:
            row_index = row['Index']
            tick_data.iloc[row_index,7] = 1
            tick_data.iloc[row_index,9] = 1
        elif count==2:
            row_index = row['Index']
            tick_data.iloc[row_index,8] = 1
            tick_data.iloc[row_index,9] = 1
        else:
            break
        count += 1
    tick_data_3 = tick_data[((tick_data['Minute_Index'] == 20) |
                             (tick_data['Minute_Index'] == 21) |
                             (tick_data['Minute_Index'] == 22) |
                             (tick_data['Minute_Index'] == 23) |
                             (tick_data['Minute_Index'] == 24) |
                             (tick_data['Minute_Index'] == 25) |
                             (tick_data['Minute_Index'] == 26) |
                             (tick_data['Minute_Index'] == 27) |
                             (tick_data['Minute_Index'] == 28) ) & (tick_data['Symbol_Index'] == stock_index)]
    count = 1
    for _ , row in tick_data_3.iterrows():
        if count==1:
            row_index = row['Index']
            tick_data.iloc[row_index,7] = 1
            tick_data.iloc[row_index,9] = 1
        elif count==2:
            row_index = row['Index']
            tick_data.iloc[row_index,8] = 1
            tick_data.iloc[row_index,9] = 1
        else:
            break
        count += 1

end = time.perf_counter()
print("The algorithm process took {:.2f} seconds.".format(end-start))

# ---------- Writer ----------

start = time.perf_counter()

print("Start writing to files...")

order_time = open(output_path, 'w')

order_time.writelines('symbol,BSflag,dataIdx,volume\n')
for _, row in tick_data.iterrows():
    if row['buyable']==1:
        sym = row['Symbol']
        letter = 'B'
        idx = row['Index']
        vol = row['trade_volume']
    elif row['sellable']==1:
        sym = row['Symbol']
        letter = 'S'
        idx = row['Index']
        vol = row['trade_volume']
    else:
        sym = row['Symbol']
        letter = 'N'
        idx = row['Index']
        vol = 0
    order_time.writelines(f'{sym},{letter},{idx},{vol}\n')
    
end = time.perf_counter()
print("The writing process took {:.2f} seconds.".format(end-start))

# ---------- Post Processing ----------

end_main = time.perf_counter()
print("Overall, the entire process took {:.2f} seconds.".format(end_main-start_main))
