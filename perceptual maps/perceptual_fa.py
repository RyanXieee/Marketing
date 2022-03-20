

from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file):                                                    
    return pd.read_csv(file, sep = ",", index_col = 0)

def order_loadings(load_mat, bases):
    load_mat_ord = pd.DataFrame(columns = range(1, len(load_mat[0]) + 1))

    for i in range(0, len(load_mat[0])):
        for j in range(0, len(load_mat)):
            if (np.argmax(abs(load_mat[j])) == i):
                load_mat_ord.loc[bases.columns[j]] = load_mat[j]
    
    return load_mat_ord

def plot_eigenvalue(eigenvalues):
    plt.figure(1)
    plt.scatter(range(1,len(eigenvalues) + 1), eigenvalues)
    plt.plot(range(1,len(eigenvalues) + 1), eigenvalues)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()


def plot_map(ax, matrix, line):
    ax.scatter(matrix.iloc[:,0],matrix.iloc[:,1])

    for i in range(0, len(matrix)):
        ax.annotate(matrix.index[i], (matrix.iloc[i,0], matrix.iloc[i,1]))
        if (line == 1):
            ax.plot([0,matrix.iloc[i,0]],[0,matrix.iloc[i,1]], linestyle="dashed",color="orange")


#load and clean data

org_data = load_data("officesun.csv")

#Determine Bases variables

bases = org_data


#Make sure factor analysis is appropriate:
#Run Bartlett test

_, p_value = calculate_bartlett_sphericity(bases)

if(p_value > 0.05):
    print("Bartlestt's test failed")


#Run KMO test

_, kmo_val = calculate_kmo(bases)

if(kmo_val < 0.6):
    print("KMO test failed")
    

#Check to see if 2 or 3 factors are appropriate:
#Run factor analysis with maximum number of factors, plot eigenvalues
    
fa = FactorAnalyzer(n_factors = len(bases.columns), rotation=None)

fa.fit(bases)

eigens, _ = fa.get_eigenvalues()

plot_eigenvalue(eigens)

num_factor = 2

#Run factor analysis with 2 or 3 factors (if appropriate)

fa = FactorAnalyzer(n_factors = num_factor, rotation="varimax")

fa.fit(bases)


#Generate 2 or 3 dimensional data

mat_2d = fa.transform(bases)

mat_2d = pd.DataFrame(mat_2d, index = bases.index)

#Plot perceptual map

fig, ax = plt.subplots()
ax.axvline(0)
ax.axhline(0)
ax.set_xlabel("dim 1")
ax.set_ylabel("dim 2")

plot_map(ax, mat_2d, 0)


#Update map with attributes and their correlation with factors
#Generate loadings matrix

load_mat = fa.loadings_

load_mat_ord = order_loadings(load_mat, bases)

#Interpret and name each factor

ax.set_xlabel("quality")
ax.set_ylabel("convenience")

plot_map(ax, load_mat_ord, 1)

fig.canvas.draw()
plt.show()
