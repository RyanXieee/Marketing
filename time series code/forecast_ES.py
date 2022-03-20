
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file):                                                    
    return pd.read_csv(file, sep = ",", header = None)



#load data

data = load_data("prod_dem.csv")
data = data.values.flatten()


#define function to forecast all exponential smoothing values given alpha

def ES(data, alpha):
    
    ES_pred = np.zeros(len(data))
    
    for period in range(0, len(data)):
        
        if (period == 0 or period == 1):
            ES_pred[period] = data[0]
        else:
            ES_pred[period] = alpha * data[period - 1] + (1 - alpha) * ES_pred[period - 1]
            
    return ES_pred




#forecast all exponential smoothing values for alpha = 0.5 and alpha = 0.9

ES_pred_alpha8 = ES(data, 0.8)
ES_pred_alpha5 = ES(data, 0.5)
ES_pred_alpha9 = ES(data, 0.9)


#plot forecasts vs ground truth

plt.figure()
fig, ax = plt.subplots(figsize=(20,5))

ax.plot(range(1, len(data) + 1), data)
ax.plot(range(1, len(data) + 1), ES_pred_alpha5)
ax.plot(range(1, len(data) + 1), ES_pred_alpha9)
ax.legend(["actual","ES alpha5", "ES alpha10"])


    
    
    
    