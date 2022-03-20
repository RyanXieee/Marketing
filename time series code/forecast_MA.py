
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file):                                                    
    return pd.read_csv(file, sep = ",", header = None)


#load data

data = load_data("prod_dem.csv")
data = data.values.flatten()


#define function for a single moving average value based on N and period to predict

def MA(data, N, perd_per):
    
    moving_avg = np.average(data[perd_per - N: perd_per])
    
    return moving_avg

#define function to forecast all possible periods after N, using previous defined function

def calc_MA_vec(data, N):
    
    MA_pred = np.zeros(len(data) - N)
    
    for period in range(0, len(data) - N):
        
        MA_pred[period] = MA(data, N, period + N)
        
    return MA_pred

#forecast all possible periods after N, for N = 5 and N = 10

MA_pred_N5 = calc_MA_vec(data, 5)
MA_pred_N10 = calc_MA_vec(data, 10)


#plot forecasts vs ground truth

plt.figure()
fig, ax = plt.subplots(figsize=(20,5))

ax.plot(range(1, len(data) + 1), data)
ax.plot(range(6, len(data) + 1), MA_pred_N5)
ax.plot(range(11, len(data) + 1), MA_pred_N10)
ax.legend(["actual","MA N5", "MA N10"])




    
    
    
    