
from sklearn import linear_model as lm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file):                                                    
    return pd.read_csv(file, sep = ",")


#Load data

dataset = load_data("officesky.csv")

#determine dependent Q(P) and independent (P) variables

obs_demand = dataset.iloc[:,1].values
price_vec = dataset.iloc[:,0].values.reshape(-1, 1)

#define and train (fit) a linear regression model for demand Q(P)

reg_model = lm.LinearRegression()
reg_model.fit(price_vec, obs_demand)


#retireve optimal parameters

a = reg_model.intercept_
b = abs(reg_model.coef_)


#calculate current profit, elasticity at current price, optimal price, and optimal profit

cost = 0.3
cur_price = 1.1

cur_profit = (cur_price - cost) * (a - b * cur_price)

elasticity = (-b * cur_price) / (a - b * cur_price)

opt_price = (a + b * cost) / (2 * b)

opt_profit = (opt_price - cost) * (a - b * opt_price)


#plot the demand curve

plt.scatter(price_vec, obs_demand)
plt.plot(price_vec, a - b * price_vec)


