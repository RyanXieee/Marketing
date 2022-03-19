
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
        

#Function to load data
def load_data(file):                                                    
    return pd.read_csv(file, sep = ",", index_col = 0)

def plot(vector, strt_per, end_per):
    plt.plot(range(strt_per, end_per + 1), vector["Period " + str(strt_per) : "Period " + str(end_per)])


#dataframe containing original number of customers by segments in first column, gross margin by segments in second column, next-priod cost by segment in third column
segments = load_data("CLV_segments.csv")
#dataframe containing transition matrix probabilities between segments
trans_matrix = load_data("CLV_transmatrix.csv")

#parameters
num_per = 20
disc_rate = 0.15

#output dataframe, contains number of customers in each segment for all periods
num_cust_mat = segments.loc[:,"Count"].to_frame()

#output dataframe, contains data on all customers by its column categories
CLV_mat= pd.DataFrame(index = ["Gross Margin", "Next-period cost", "Net Margin", "Discount Factor", "Discounted Net Margin", "Cumulated Disc Net Margin"])

#array of customer numbers for current period (input data)
cur_per_num_cust = num_cust_mat.to_numpy()
#cur_per_num_cust = np.array([0,1,0,0])


#comulative discounted net margin (CLV)
CLV = 0

for per in range(1, num_per + 1):
    
    #array of customer numbers for next period (to be calculated below)
    next_per_num_cust = np.zeros(len(trans_matrix))
    
    for i in range(0,len(trans_matrix)):
        for j in range(0,len(trans_matrix)):
            next_per_num_cust[i] += trans_matrix.iloc[j,i] * cur_per_num_cust[j] / 100
    
    per_name = "Period " + str(per)
    
    #adding calculated number of customers for next period to our output dataframe
    num_cust_mat[per_name] = pd.Series(next_per_num_cust, num_cust_mat.index)
    
    #array of customer data for next period (to be calculated)
    next_per_CLV_data = np.zeros(len(CLV_mat))
    
    for i in range(0, len(next_per_num_cust)):
        next_per_CLV_data[0] += next_per_num_cust[i] * segments.iloc[i, 1]          #gross margin (from number of customers we end up with, i.e., next period)
        next_per_CLV_data[1] += cur_per_num_cust[i] * segments.iloc[i, 2]           #next-period cost (from number of customers we advertised to, i.e., current period)

    next_per_CLV_data[2] = next_per_CLV_data[0] - next_per_CLV_data[1]              #net margin
    next_per_CLV_data[3] = pow(1/(1+disc_rate), per)                                #discount factor
    next_per_CLV_data[4] = next_per_CLV_data[2] * next_per_CLV_data[3]              #discounted net margin
    
    CLV += next_per_CLV_data[4]                                       #cumul disc margin
    
    next_per_CLV_data[5] = CLV                                        #CLV
    
    #adding calculated customer CLV data for next period to our output dataframe
    CLV_mat[per_name] = pd.Series(next_per_CLV_data, CLV_mat.index)
    
    #changing current period for next loop
    cur_per_num_cust = next_per_num_cust
    
plot(num_cust_mat.loc["Active customers" , :], 1, 10)
plot(num_cust_mat.loc["Warm customers" , :], 1, 10)
plot(num_cust_mat.loc["Cold customers" , :], 1, 10)
plot(num_cust_mat.loc["Lost customers" , :], 1, 10)
    