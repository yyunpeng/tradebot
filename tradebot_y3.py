# -*- coding: utf-8 -*-
"""
Created on Sun Aug  7 17:59:56 2022

@author: xuyun
"""


#%% libraries
import numpy as np
import time
import statsmodels.api as sm
import yfinance as yf
import pandas as pd
from random import randrange
from pathlib import Path  
from ib_insync import *

#%%
print('Socket connection established')
client_id = 15
port_paper = 7496
ib_conn = IB()
ib_conn.connect(host='127.0.0.1', port=port_paper, clientId=client_id)

#%% get pos_size
print('Tickers loaded')
print('Total asset updated')

stocks = ['GOOGL','AMZN','AAPL','MSFT','TSLA','XOM','META','V','NVDA','JNJ','PG','WMT']
totalAsset = 100000

#%% Strategy II
print('Strategy loaded')

def macd(dataframe, slow, fast, window): 
    df_temp = dataframe.copy()
    df_temp['ma_fast'] = df_temp['Adj Close'].ewm(span=fast, min_periods=fast).mean()
    df_temp['ma_slow'] = df_temp['Adj Close'].ewm(span=slow, min_periods=slow).mean()
    df_temp['macd'] = df_temp['ma_fast'] - df_temp['ma_slow']
    df_temp['signal'] = df_temp['macd'].ewm(span=window, min_periods=window).mean()
    return df_temp.loc[:,['macd','signal']]

def bb(dataframe2,window2):
    df_temp2 = dataframe2.copy()
    df_temp2['middle_band'] = df_temp2['Adj Close'].rolling(window2).mean()
    df_temp2['upper_band'] = df_temp2['middle_band'] + 2*df_temp2['Adj Close'].rolling(window2).std(ddof=1)
    df_temp2['lower_band'] = df_temp2['middle_band'] - 2*df_temp2['Adj Close'].rolling(window2).std(ddof=1)
    df_temp2['band_width'] = df_temp2['upper_band'] - df_temp2['lower_band']
    return df_temp2.loc[:,['middle_band','upper_band','lower_band','band_width']]

def atr(dataframe3, window3):
    df_temp3 = dataframe3.copy()
    df_temp3['high-low'] = df_temp3['High'] - df_temp3['Low']
    df_temp3['high-preious_low'] = df_temp3['High'] - df_temp3['Adj Close'].shift(1)
    df_temp3['low-preious_low'] = df_temp3['Low'] - df_temp3['Adj Close'].shift(1)
    df_temp3['True_Range'] = df_temp3[['high-low','high-preious_low','low-preious_low']].max(axis=1, skipna=False)
    df_temp3['ATR'] = df_temp3['True_Range'].ewm(com=window3,min_periods=window3).mean()
    return df_temp3['ATR'] 

def rsi(dataframe4, window4):
    df_temp4 = dataframe4.copy()
    df_temp4['index'] = range(len(df_temp4))
    df_temp4['gain'] = [0]*len(df_temp4)
    df_temp4['loss'] = [0]*len(df_temp4)
    count=0    
    for k1 in df_temp4['Adj Close'] - df_temp4['Adj Close'].shift(1):     
        if k1>=0: 
            df_temp4['loss'].loc[df_temp4['index']==count] = 0
            df_temp4['gain'].loc[df_temp4['index']==count] = df_temp4['Adj Close'] - df_temp4['Adj Close'].shift(1)
        else: 
            df_temp4['loss'].loc[df_temp4['index']==count] = df_temp4['Adj Close'].shift(1) - df_temp4['Adj Close'] 
            df_temp4['gain'].loc[df_temp4['index']==count] = 0
        count+=1    
    df_temp4['avg_gain'] = [0]*len(df_temp4)
    df_temp4['avg_loss'] = [0]*len(df_temp4)      
    for idx in df_temp4['index']: 
        if idx<(window4-1): 
            df_temp4['avg_gain'].loc[df_temp4['index']==idx] = 0 
            df_temp4['avg_loss'].loc[df_temp4['index']==idx] = 0 
        elif idx==window4-1:
            df_temp4['avg_gain'].loc[df_temp4['index']==idx] = df_temp4['gain'][0:window4-1].mean()
            df_temp4['avg_loss'].loc[df_temp4['index']==idx] = df_temp4['loss'][0:window4-1].mean()
        else: 
            df_temp4['avg_gain'].loc[df_temp4['index']==idx] = (df_temp4['avg_gain'].loc[df_temp4['index']==(idx-1)][0] 
                                                                * (window4-1) + df_temp4['gain'].loc[df_temp4['index']==idx][0]) /window4
            df_temp4['avg_loss'].loc[df_temp4['index']==idx] = (df_temp4['avg_loss'].loc[df_temp4['index']==(idx-1)][0] 
                                                                * (window4-1) + df_temp4['loss'].loc[df_temp4['index']==idx][0]) /window4    
    df_temp4['RSI'] = 100-(100/(1+df_temp4['avg_gain']/df_temp4['avg_loss']))
    return df_temp4['RSI']

def adx(dataframe5, window5):
    df_temp5 = dataframe5.copy()
    df_temp5['ATR'] = atr(dataframe5, window5)
    df_temp5['upmove'] = df_temp5['High'] - df_temp5['High'].shift(1)
    df_temp5['downmove'] = df_temp5['Low'].shift(1) - df_temp5['Low']
    df_temp5['+dm'] = np.where((df_temp5['upmove']>df_temp5['downmove']) & (df_temp5['upmove']>0), df_temp5['upmove'], 0)
    df_temp5['-dm'] = np.where((df_temp5['upmove']<df_temp5['downmove']) & (df_temp5['downmove']>0), df_temp5['downmove'], 0)
    df_temp5['+di'] = 100*(df_temp5['+dm']/df_temp5['ATR']).ewm(com=window5, min_periods=window5).mean()
    df_temp5['-di'] = 100*(df_temp5['-dm']/df_temp5['ATR']).ewm(com=window5, min_periods=window5).mean()
    df_temp5['ADX'] = 100* abs( (df_temp5['+di']-df_temp5['-di']) / 
                               (df_temp5['+di']+df_temp5['-di'])).ewm(com=window5, min_periods=window5).mean()
    return df_temp5['ADX']

def merge_many(DF):
    merged_df = DF
    merged_df["Date"] = merged_df.index
    merged_df[['MACD','SIGNAL']] = macd(merged_df,26,12,9)
    merged_df[['Middle_Band','Upper_Band','Lower_Band','Band_Width']] = bb(merged_df,12)
    merged_df['ATR'] = atr(merged_df,14)
    merged_df['RSI'] = rsi(merged_df,14)
    merged_df['ADX'] = adx(merged_df,20)
    return merged_df

def decision_test(new_data_df, stock_ticker):
    buyPrices_path = 'D:\\a.NTU\\Y2 summer\\trading bot\\trade bot\\IB tests\\stockPrices\\buyPrices\\{}.csv'.format(stock_ticker)
    sellPrices_path = 'D:\\a.NTU\\Y2 summer\\trading bot\\trade bot\\IB tests\\stockPrices\\sellPrices\\{}.csv'.format(stock_ticker)
    Buy_df = pd.read_csv(buyPrices_path)
    Sell_df = pd.read_csv(sellPrices_path)
    signal = ''
    df = new_data_df
    if df['MACD'].tolist()[-1]>df['SIGNAL'].tolist()[-1] and df['ADX'].tolist()[-1]<=40 and df['RSI'].tolist()[-1]<=30 and df['Lower_Band'].tolist()[-1]*0.5<df['Adj Close'].tolist()[-1]<df['Lower_Band'].tolist()[-1]*1.3:
        if Buy_df.iloc[0,1]==0: 
            signal = 'Buy'
            
    elif Sell_df.iloc[0,1] != 0 and new_data_df['Adj Close'][-1] < 0.95*Sell_df.iloc[0,1]:
        signal = 'Close_buy'
        
    elif df['MACD'].tolist()[1]<df['SIGNAL'].tolist()[-1] and df['ADX'].tolist()[-1]<=40 and df['RSI'].tolist()[-1]>=70 and df['Upper_Band'].tolist()[-1]*0.7<df['Adj Close'].tolist()[-1]<df['Upper_Band'].tolist()[-1]*1.5 :
        if Sell_df.iloc[0,1]==0:
            signal = 'Sell'
            
    elif Buy_df.iloc[0,1] != 0 and new_data_df['Adj Close'][-1] > 1.3*Buy_df.iloc[0,1]:
        signal = 'Close_sell'
            
    return signal


#%% Simplify multi-threading & placing order
print('Multi-threading and order execution simplified')

def write_0(the_file_path): 
    price_df = pd.DataFrame({0})
    filepath = Path(the_file_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    price_df.to_csv(filepath)

def write_price(the_each_df2, the_file_path):
    price_df = pd.DataFrame({the_each_df2['Adj Close'][-1]})
    filepath = Path(the_file_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    price_df.to_csv(filepath)

def write_posSize(pos_size, the_ticker, the_file_path): 
    stkInfo = yf.download(the_ticker, period='5d', interval='1m')
    stkInfo_0 = pd.DataFrame(stkInfo)
    stkInfo_0.dropna(how='any', inplace=True)
    pos_df = pd.DataFrame({pos_size})
    filepath = Path(the_file_path)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pos_df.to_csv(filepath)

def exe_order(opt_conn, the_pos_size, the_ticker, the_trade_signal): 
    contract_obj = Stock(the_ticker, 'SMART', 'USD')
    order_obj = MarketOrder(the_trade_signal, the_pos_size)
    trade = opt_conn.placeOrder(contract_obj, order_obj)

#%% write_0_to_restart
'''
for j in stocks: 
    buyPrices_path = 'D:\\a.NTU\\Y2 summer\\trading bot\\trade bot\\IB tests\\stockPrices\\buyPrices\\{}.csv'.format(j)
    sellPrices_path = 'D:\\a.NTU\\Y2 summer\\trading bot\\trade bot\\IB tests\\stockPrices\\sellPrices\\{}.csv'.format(j)
    posSize_path = 'D:\\a.NTU\\Y2 summer\\trading bot\\trade bot\\IB tests\\stockPrices\\positionSize\\{}.csv'.format(j)
    write_0(buyPrices_path)
    write_0(sellPrices_path)
    write_0(posSize_path)
'''

#%% main

r=11

for i in range(r):
    try:
        for each_stock in stocks:
            temp_0 = yf.download(each_stock, period='1mo', interval='5m')
            temp = pd.DataFrame(temp_0)
            temp.dropna(how='any', inplace=True)
            each_df1 = temp
            each_df2 = merge_many(each_df1)
            trade_signal = decision_test(each_df2, each_stock)
            
            buyPrices_path = 'D:\\a.NTU\\Y2 summer\\trading bot\\trade bot\\IB tests\\stockPrices\\buyPrices\\{}.csv'.format(each_stock)
            sellPrices_path = 'D:\\a.NTU\\Y2 summer\\trading bot\\trade bot\\IB tests\\stockPrices\\sellPrices\\{}.csv'.format(each_stock)
            posSize_path = 'D:\\a.NTU\\Y2 summer\\trading bot\\trade bot\\IB tests\\stockPrices\\positionSize\\{}.csv'.format(each_stock)
            Buy_df = pd.read_csv(buyPrices_path)
            Sell_df = pd.read_csv(sellPrices_path)
            pos_df = pd.read_csv(posSize_path)
            
            if trade_signal == "Buy":
                pos_size = int((0.5*totalAsset)//len(stocks)//each_df2['Adj Close'][-1])
                if pos_size>500 or pos_size<20: 
                    pos_size = 20
                exe_order(ib_conn, pos_size, each_stock, 'BUY')
                write_posSize(pos_size, each_stock, posSize_path)
                write_price(each_df2, buyPrices_path)
                print("New long position:", each_stock)
                time.sleep(2)

            elif trade_signal == 'Close_buy':
                pos_size = pos_df.iloc[0,1]
                exe_order(ib_conn, pos_size, each_stock, 'BUY')
                write_0(sellPrices_path)
                write_0(posSize_path)
                print('Buying back', each_stock, 'to close a short position')
                time.sleep(2)

            elif trade_signal == "Sell":
                pos_size = int((0.5*totalAsset)//len(stocks)//each_df2['Adj Close'][-1])
                if pos_size>500 or pos_size<20: 
                    pos_size = 20                
                exe_order(ib_conn, pos_size, each_stock, 'SELL')
                write_posSize(pos_size, each_stock, posSize_path)
                write_price(each_df2, sellPrices_path)
                print("New short position:", each_stock)
                time.sleep(2)
                
            elif trade_signal == "Close_sell":
                exe_order(ib_conn, pos_size, each_stock, 'SELL')
                write_0(buyPrices_path)
                write_0(posSize_path)
                print('Selling', each_stock, 'to close a long position')
                time.sleep(2)

            elif trade_signal == '': 
                print('No signals, skip')
                time.sleep(3)
            
        time.sleep(4.7*60)

    except KeyboardInterrupt or ValueError: 
        print('''
              ------------------------------
              Exception received. Exiting...
              ------------------------------
              ''')
        exit()


print(''' 
      ------------------------------
            Test completed. 
      ------------------------------
      ''')

ib_conn.disconnect()
time.sleep(3)
print('Disconnected')
