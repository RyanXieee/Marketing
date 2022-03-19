
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering as Aggl
from sklearn.cluster import KMeans


def load_data(file):                                                    
    return pd.read_csv(file, sep = ",", index_col = False)

def plot_eigenvalue(eigenvalues):
    plt.figure(1)
    plt.scatter(range(1,len(eigenvalues) + 1), eigenvalues)
    plt.plot(range(1,len(eigenvalues) + 1), eigenvalues)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()
    
    
def order_loadings(load_mat, bases):
    load_mat_ord = pd.DataFrame(columns = range(1, len(load_mat[0]) + 1))

    for i in range(0, len(load_mat[0])):
        for j in range(0, len(load_mat)):
            if (np.argmax(abs(load_mat[j])) == i):
                load_mat_ord.loc[bases.columns[j]] = load_mat[j]
    
    return load_mat_ord


#Load data & clean it

org_data = load_data("toothpaste.csv")
org_data = org_data.drop("id", axis = 1)

#Determine Bases variables, and non-bases variables 

demog = org_data.iloc[:,6:]
bases = org_data.iloc[:,0:6]


#Make sure factor analysis is appropriate:
#Run Bartlett test

_, p_value = calculate_bartlett_sphericity(bases)

if(p_value > 0.05):
    print("Bartlestt's test failed")


#Run KMO test

_, kmo_val = calculate_kmo(bases)

if(kmo_val < 0.6):
    print("KMO test failed")
    

#Determine number of factors:
#Run factor analysis with maximum number of factors, plot eigenvalues
    
fa = FactorAnalyzer(n_factors = len(bases.columns), rotation=None)

fa.fit(bases)

eigens, _ = fa.get_eigenvalues()

plot_eigenvalue(eigens)

n_factors = 2


#Run final factor analysis with number of factors from eigenvalue plot

fa_final = FactorAnalyzer(n_factors = n_factors, rotation="varimax")


#Get reduced data matrix & Loadings matrix

red_mat = fa_final.fit_transform(bases)

load_mat = fa_final.loadings_

load_mat_ord = order_loadings(load_mat, bases)

#Determine which factors load with which questions


#interpret and name each factor

factor_names = ["health benefits", "social benefits"]

#Use hierarchical clustering to get number of clusters

clust_model_hier = Aggl(linkage='ward')
clust_model_hier.fit(red_mat)

#Plot dendrogram and analyze number of clusters

dendrogram(linkage(red_mat, method ='ward'))

num_clust = 3

#Use K-means to get final customer segments

clust_model_kmeans = KMeans(n_clusters= num_clust, random_state = 0)
kmeans_fitted = clust_model_kmeans.fit(red_mat)


#print cluster centers and labels

customer_centers = pd.DataFrame(kmeans_fitted.cluster_centers_, columns = factor_names)


#Use cluster centers to name customer segments

cluster_names = ["Health seekers", "Lost souls", "Social conscious"]

customer_segment_assignment = kmeans_fitted.labels_

#print cluster demographics

for cluster in range(0, num_clust):
    print(demog.iloc[customer_segment_assignment == cluster, :].describe())


#plot cluster differences based on factors

plt.figure(2)
for cluster in range(0, num_clust):
    plt.plot(factor_names, customer_centers.iloc[cluster, :], marker='o', label = cluster_names[cluster])
plt.legend()

