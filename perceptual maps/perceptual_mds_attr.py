# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.manifold import MDS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(file):                                                    
    return pd.read_csv(file, sep = ",", index_col = 0)

def plot_stress(stress):
    plt.figure(1)
    plt.scatter(range(1,len(stress) + 1), stress)
    plt.plot(range(1,len(stress) + 1), stress)
    plt.xlabel('Dimensions')
    plt.ylabel('Stress')
    plt.ylim((0,5))
    plt.grid()
    plt.show()
    
def plot_map(ax, matrix):
    ax.scatter(matrix.iloc[:,0],matrix.iloc[:,1])

    for i in range(0, len(matrix)):
        ax.annotate(matrix.index[i], (matrix.iloc[i,0], matrix.iloc[i,1]))


#load and clean data

org_data = load_data("officesun.csv")


#Determine Bases variables

bases = org_data

#Create and fit MDS model

mds_model = MDS(n_components= 2, random_state = 0)

mat_2d = mds_model.fit_transform(bases)

mat_2d = pd.DataFrame(mat_2d, index = bases.index)


#Plot perceptual map

fig, ax = plt.subplots()
ax.axvline(0)
ax.axhline(0)
ax.set_xlabel("dim 1")
ax.set_ylabel("dim 2")

plot_map(ax, mat_2d)
      
dis_mat = pd.DataFrame(mds_model.dissimilarity_matrix_, index = bases.index, columns = bases.index)
