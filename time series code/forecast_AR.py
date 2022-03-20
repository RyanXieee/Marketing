
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg


def load_data(file):                                                    
    return pd.read_csv(file, sep = ",", header = None)


#load data

data = load_data("prod_dem.csv")
data = data.values.flatten()

#define function to train an Autoregression model given p

def AR(data, p):
    
    AR_model = AutoReg(data, lags = p, old_names = False)
    
    AR_model_trained = AR_model.fit()
    
    print(AR_model_trained.summary())
    
    AR_pred = AR_model_trained.predict(p, len(data) + 5 - 1)
    
    return AR_pred

#forecast all autoregression values after p = 5 and p = 10

AR_pred_p3 = AR(data[:9], 3)
AR_pred_p5 = AR(data[:-5], 5)
AR_pred_p10 = AR(data[:-5], 10)


#plot forecasts vs ground truth

plt.figure()
fig, ax = plt.subplots(figsize=(20,5))

ax.plot(range(1, len(data) + 1), data)
ax.plot(range(6, len(data) + 1), AR_pred_p5)
ax.plot(range(11, len(data) + 1), AR_pred_p10)
ax.legend(["actual","AR p5", "AR p10"])


    
    
    
    