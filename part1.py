import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering, KMeans
import pickle
import utils as u


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(data,labels,num_of_clusters,random_init=42):
    kmm=KMeans(n_clusters=num_of_clusters,random_state=random_init)
    ss=StandardScaler()
    train=ss.fit_transform(data)
    kmm.fit(train,labels)
    preds=kmm.predict(train)
    return preds


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    numberOfSamples = 100
    randomState = 42

    noisyCircles = datasets.make_circles(n_samples=numberOfSamples, factor=0.5, noise=0.05, random_state=randomState)
    noisyMoons = datasets.make_moons(n_samples=numberOfSamples, noise = 0.05, random_state=randomState)
    blobs = datasets.make_blobs(n_samples=numberOfSamples, random_state=randomState)
    variedBlobs = datasets.make_blobs(n_samples=numberOfSamples, cluster_std=[1.0,2.5,0.5], random_state=randomState)
    X, y = datasets.make_blobs(n_samples=numberOfSamples, random_state=randomState)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    dct = answers["1A: datasets"] = {
        "nc" : noisyCircles,
        "nm" : noisyMoons,
        "bvv" : variedBlobs,
        "add" : aniso,
        "b" : blobs
    }
    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans
    results_from_fit=dct


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    
    Kmeans_dict_plotting={}
    for dataset_i in answers['1A: datasets'].keys():
        acc=[]
        dataset_cluster={}
        for num_cluster in [2,3,5,10]:
            #x_y=answers['1A: datasets'][dataset_i]
            preds=dct(answers['1A: datasets'][dataset_i][0],answers['1A: datasets'][dataset_i][1],num_cluster,42)
            dataset_cluster[num_cluster]=preds
            #accc=accuracy_score(preds,answers["1A: datasets"][dataset_i][1])
            #Kmeans_dict_plotting[dataset_i]=accc
            #acc.append([x_y,preds])
        acc.append((answers['1A: datasets'][dataset_i][0],answers['1A: datasets'][dataset_i][1]))
        acc.append(dataset_cluster)
        Kmeans_dict_plotting[dataset_i]=acc

    myplt.plot_part1C(Kmeans_dict_plotting,'part1Question3.jpg')

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    
    
    dct=answers["1C: cluster successes"] = {"bvv": [3], "add": [3],"b":[3]} 

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)

    dct=answers["1C: cluster failures"] = ["nc","nm"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.

    Kmeans_dict_plotting={}
    for dataset_i in answers['1A: datasets'].keys():
        acc=[]
        dataset_cluster={}
        for num_cluster in [2,3]:
            preds=results_from_fit(answers['1A: datasets'][dataset_i][0],answers['1A: datasets'][dataset_i][1],num_cluster,42)
            dataset_cluster[num_cluster]=preds
        acc.append((answers['1A: datasets'][dataset_i][0],answers['1A: datasets'][dataset_i][1]))
        acc.append(dataset_cluster)
        Kmeans_dict_plotting[dataset_i]=acc
    myplt.plot_part1C(Kmeans_dict_plotting,'part1Question4.jpg')

    dct = answers["1D: datasets sensitive to initialization"] = ["nc","nm"]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
