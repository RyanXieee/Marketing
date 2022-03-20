

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


def load_data(file):                                                    
    return pd.read_csv(file, sep = ",")


#Load data

dataset = load_data("officesky.csv")

#determine dependent Q(P) and independent (P) variables

obs_demand = dataset.iloc[:,1].values
price_vec = dataset.iloc[:,0].values

#define function  for Q(P) that gives demand given parameters a, b

def Qp(params, price_vec):
    
    a = params[0]
    b = params[1]
    
    Q_vec = a / (price_vec**b) 
    
    return Q_vec



#develop optimization framework to train (fit) constant elasticity demand curve
# initialize vector of parameters [a,b] (guess)

params_guess = [1,1]


#define function that gives the error of the demand curve prediction (with provided parameters) vs the ground observed truth

def residual(params, price_vec, obs_demand):
    
    Q_pred = Qp(params, price_vec)
    
    error = Q_pred - obs_demand
    
    return error

                 
#model and optimize the developed error function using nonlinear least squares
 
params_opt, success = leastsq(residual, params_guess, args=(price_vec, obs_demand))


#retireve optimal parameters

a_opt = params_opt[0]
b_opt = params_opt[1]


#calculate current profit, optimal price, and optimal profit

cost = 0.3

cur_price = 1.1

cur_profit = (cur_price - cost) * Qp(params_opt, cur_price)

opt_price = a_opt * b_opt * cost / (a_opt *b_opt - a_opt)

opt_profit = (opt_price - cost) * Qp(params_opt, opt_price)


#plot the demand curve

price_vec2 = np.linspace(0.5,2, num=100)

plt.scatter(price_vec, obs_demand)
plt.plot(price_vec2, Qp(params_opt, price_vec2))


