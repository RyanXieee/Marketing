

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


def load_data(file):                                                    
    return pd.read_csv(file, sep = ",", index_col = False, header = None)



#load data

obs_sales = load_data("officemoon.csv")
obs_sales = obs_sales.values.flatten()

t_vec = np.linspace(1, 10, num = 10)

#define function  for n(t) that gives bass model forecast given parameters N, p, q

def nt(bass_param, t_vec):
    
    N = bass_param[0]
    p = bass_param[1]
    q = bass_param[2]
    
    nt_vec = N * p * (p + q)**2 * np.exp(-(p+q) * t_vec) / (p + q * np.exp(-(p+q) * t_vec))**2
    
    return nt_vec


#develop optimization framework to predict Bass model parameters
# initialize vector of bass parameters [N,p,q] (guess)

param_guess = [50,0.1,0.1]


#define function that gives the error of the bass model prediction (with provided parameters) vs the ground observed truth

def error_fun(bass_param, t_vec, obs_sales):
    
    nt_pred_vec = nt(bass_param, t_vec)
    
    error = nt_pred_vec - obs_sales
    
    return error
                 
#model and optimize the developed error function using nonlinear least squares
 
param_opt, success = leastsq(error_fun, param_guess, args = (t_vec, obs_sales))


#retireve optimal bass parameters

N_opt = param_opt[0]
p_opt = param_opt[1]
q_opt = param_opt[2]


#plot the optimized bass model for n(t)

t_forecast = np.linspace(1, 30, num = 30)
nt_forecast =nt(param_opt, t_forecast)

plt.figure()
plt.plot(t_forecast, nt_forecast)

#plot the optimized bass model for N(t)

Nt_forecast = np.cumsum(nt_forecast)

plt.figure()
plt.plot(t_forecast, Nt_forecast)
