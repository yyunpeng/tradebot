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
tick_data['buyable'] = [0] * len(tick_data)
tick_data['sellable'] = [0] * len(tick_data)
tick_data['trade_volume'] = [0] * len(tick_data)

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
arr = np.empty((100, 240))
arr[0:100, 0:240] = np.nan

for t in range(30, 239):
    start_simulate = time.perf_counter()
    dim, mean_matrix_list, covariance_matrix_list = simulator.execute(t=t, tau_limit=5, r=0.1, path_num=5) # path_num = 200 will take 17.87 seconds, not too bad
    end_simulate = time.perf_counter()
    print("Simulation at Minute_Index = '{}' took {:.2f} seconds.".format(t, end_simulate-start_simulate))
    start_algo = time.perf_counter()
    for stock_index in range(100):
        start_algo = time.perf_counter()
        algo_checker = Algo_Check(E_S=mean_matrix_list[stock_index], 
                                  Sigma=covariance_matrix_list[stock_index], 
                                  dim=dim, 
                                  number_of_share=50, 
                                  number_of_iterations=2) # Number of iterations for minimization. Default is 1000 but will take 4-7 seconds which is too long. Restricting to 25 iterations, each iteration only takes 0.14-0.16 seconds which is much better.
        x_estimate = algo_checker.execute()
        # print('Stock: {}; x_estimate: {}'.format(stock_index, x_estimate))
        
        del algo_checker
        
        arr[stock_index,t] = x_estimate[0]
        # x_estimate

    end_algo = time.perf_counter()
    print("Running algorithm at Minute_Index = '{}' took {:.2f} seconds.".format(t, end_algo-start_algo)) # Algorithm takes 33 seconds, not too bad

    del mean_matrix_list, covariance_matrix_list

arr = -arr

for i in range(100):
    # Ignore first 30 entries for normalization
    extracted_array_original = arr[i,30:-1]
    row_sum = np.nansum(extracted_array_original)
    normalization_factor = 97 / row_sum
    extracted_array_normalised = normalization_factor * extracted_array_original
    extracted_array_integer = np.floor(extracted_array_normalised).astype(int)
    difference = 97 - int(np.nansum(extracted_array_integer))
    extracted_array_after_decimal = extracted_array_normalised - extracted_array_integer
    # print(difference)
    # print(extracted_array_integer)
    # print(extracted_array_after_decimal)
    index_list = np.argpartition(extracted_array_after_decimal, -difference)[-difference:]
    for j in range(extracted_array_integer.shape[0]):
        if j in index_list:
            extracted_array_integer[j] +=1
    # print(np.nansum(extracted_array_integer))
    arr[i,30:-1] = extracted_array_integer 

arr[:,239] = 0
arr[:,0:30] = 0
arr = np.floor(arr).astype(int)

print(arr.shape)
print(arr)
# print(arr[0,:])
# print(np.nansum(arr[0,:]))
# print(arr[1,:])
# print(np.nansum(arr[1,:]))
# print(arr[99,:])
# print(np.nansum(arr[99,:]))


tick_list = []
for i in range(100):
    each_stock = []
    for j in range(240):
        each_stock.append([])
    tick_list.append(each_stock)

for stock_index in range(100): 
    for minute_index in range(240): 
        if arr[stock_index][minute_index]== 0:
            pass
        else: 
            tick_data_per_minute = tick_data[(tick_data['Minute_Index'] == minute_index) & (tick_data['Symbol_Index'] == stock_index)]
            for _ , row in tick_data_per_minute.iterrows():
                tup_index = row['Index']
                tup_price = row['Price']
                tup_tick_idx = row['Tick_Index']
                tup_volume = arr[stock_index,minute_index]
                tup = (tup_index,tup_tick_idx,tup_price,tup_volume)
                tick_list[stock_index][minute_index].append(tup)

print(tick_list[50][2])
print(tick_list[99][30])
print(tick_list[99][31])
print(tick_list[99][32])
print(tick_list[99][33])
print(tick_list[99][34])
print(tick_list[99][35])
print(tick_list[99][36])
print(tick_list[99][37])
print(tick_list[99][38])

def dfs(stock_index, m,n,earliest_second):
    if (n-m)==1:
        output_tuple = ()
        max_profit = 0
        for tup in tick_list[stock_index][m]:
            if (tup[1] > (earliest_second+60)) and (tup[2] * tup[3] > max_profit):
                output_tuple = tup
                max_profit = tup[2] * tup[3]
        return max_profit, output_tuple
    else:
        first_small_list = []
        for tup in tick_list[stock_index][m]:
            if tup[1] > (earliest_second + 60):
                first_small_list.append(tup)
        max_profit, output_tuple = dfs(stock_index, m+1,n,earliest_second)
        for tup in first_small_list:
            profit_temp, output_tuple_temp = dfs(stock_index, m+1,n,tup[1])
            if (profit_temp + tup[2] * tup[3]) > max_profit:
                max_profit = profit_temp + tup[2] * tup[3]
                output_tuple = tup + output_tuple_temp
        return max_profit, output_tuple

for stock_index in range(99,100):
    for i in range(30,39):
        if tick_list[stock_index][i]==[]:
            pass
        else:
            j = i+1
            while True:
                if j==240 or tick_list[stock_index][j]==[]:
                    break
                else:
                    j += 1
            if (j-i)==1:
                # Do something
                max_profit = 0
                max_tup = ()
                for tup in tick_list[stock_index][i]:
                    # print(tup)
                    if tup[2] * tup[3] > max_profit:
                        max_profit = tup[2] * tup[3]
                        max_tup = tup
                tick_data.iloc[max_tup[0],8] = 1
                tick_data.iloc[max_tup[0],9] = max_tup[3]
                # print(max_tup)
                # print(tick_data.iloc[max_tup[0]-1:max_tup[0]+2,:])
            else:
                # Do something
                max_profit, output_tuple = dfs(stock_index=stock_index, m=i,n=j,earliest_second=0)
                print(max_profit, output_tuple)
                for k in range(0,len(output_tuple), 4):
                    tup_0 = output_tuple[k]
                    tup_1 = output_tuple[k+1]
                    tup_2 = output_tuple[k+2]
                    tup_3 = output_tuple[k+3]
                    print(tup_0,tup_1,tup_2,tup_3)
                    tick_data.iloc[tup_0,8] = 1
                    tick_data.iloc[tup_0,9] = tup_3
                    print(tick_data.iloc[tup_0-1:tup_0+2,:])
            

# indexes = []
# for stock_list in range(100): 
#     temp_list = tick_list[stock_list]
#     for i in range(len(temp_list)):

#         if temp_list[i] and ((i == 0) or (not temp_list[i-1])) and ((i == len(temp_list)-1) or (not temp_list[i+1])):
#             max_val = -float('inf')
#             max_index = -1
#             for j in range(len(temp_list[i])):
#                 if temp_list[i][j][2] > max_val:
#                     max_val = temp_list[i][j][2]
#                     max_index = j
#             if max_index != -1:
#                 indexes.append(temp_list[i][max_index][1])

# tick_data['Trade_or_not'] = tick_data['Tick_Code'].apply(lambda x: get_ms(x))

end = time.perf_counter()
print("The algorithm process took {:.2f} seconds.".format(end-start))


# ---------- Writer ----------

# order_time.writelines('symbol,BSflag,dataIdx,volume\n')
# for _, row in tick_data.iterrows():
#     if row['buyable']==1:
#         order_time.writelines(f'{sym},{order[0]},{nTick},{order[1]}\n')
#     elif row['sellable']==1:
        


# ---------- Post Processing ----------

end_main = time.perf_counter()
print("Overall, the entire process took {:.2f} seconds.".format(end_main-start_main))
