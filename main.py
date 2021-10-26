import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

aggregation_ds = pd.read_csv('./Donnees_projet_2021/aggregation.txt', sep="\t")
jain_ds = pd.read_csv('./Donnees_projet_2021/jain.txt', sep="\t")
pathbased_ds = pd.read_csv('./Donnees_projet_2021/pathbased.txt', sep="\t")

aggregation_df = aggregation_ds.to_numpy()
jain_df = jain_ds.to_numpy()
pathbased_df = pathbased_ds.to_numpy()

int_to_color = {1: 'blue', 2:'red', 3:'green', 4: 'yellow', 5: 'orange', 6: 'purple', 7: 'grey'}



# draw points for each dataset with one color per cluster
def draw_scatter():
    fig, axs = plt.subplots(2, 2)
    for point in aggregation_df:
        color = int_to_color[point[2]]
        axs[0, 0].scatter(point[0], point[1], c=color)

    axs[0, 0].set_title('Aggregation')

    for point in jain_df:
        color = int_to_color[point[2]]
        axs[0, 1].scatter(point[0], point[1], c=color)

    axs[0, 1].set_title('Jain')

    for point in pathbased_df:
        color = int_to_color[point[2]]
        axs[1, 0].scatter(point[0], point[1], c=color)

    axs[1, 0].set_title('Pathbased')

    plt.show()
    return;

# comput Kmean algorithm and draw clusters
def kmeans_clusters():
    fig, axs = plt.subplots(3, 2)
    # Aggregation
    km_aggregation = KMeans(n_clusters=7, random_state=42, n_init=100)
    km_aggregation.fit_predict(aggregation_df)
    i = 0
    for point in aggregation_df:
        color = int_to_color[km_aggregation.labels_[i] + 1]
        axs[0, 0].scatter(point[0], point[1], c=color)
        i += 1

    centroids = km_aggregation.cluster_centers_

    for center in centroids:
        axs[0, 0].scatter(center[0], center[1], c="black")


    axs[0, 0].set_title('Aggregation with Kmeans')

    # Jain
    km_jain = KMeans(n_clusters=2, random_state=42, n_init=100)
    km_jain.fit_predict(jain_df)
    i = 0
    for point in jain_df:
        color = int_to_color[km_jain.labels_[i] + 1]
        axs[1, 0].scatter(point[0], point[1], c=color)
        i += 1

    centroids = km_jain.cluster_centers_

    for center in centroids:
        axs[1, 0].scatter(center[0], center[1], c="black")

    axs[1, 0].set_title('Jain with Kmeans')

    # pathbased
    km_pathbased = KMeans(n_clusters=3, random_state=42, n_init=100)
    km_pathbased.fit_predict(pathbased_df)
    i = 0
    for point in pathbased_df:
        color = int_to_color[km_pathbased.labels_[i] + 1]
        axs[2, 0].scatter(point[0], point[1], c=color)
        i += 1

    centroids = km_pathbased.cluster_centers_

    for center in centroids:
        axs[2, 0].scatter(center[0], center[1], c="black")

    axs[2, 0].set_title('Pathbased with Kmeans')


    for point in aggregation_df:
        color = int_to_color[point[2]]
        axs[0, 1].scatter(point[0], point[1], c=color)

    axs[0, 1].set_title('Aggregation')

    for point in jain_df:
        color = int_to_color[point[2]]
        axs[1, 1].scatter(point[0], point[1], c=color)

    axs[1, 1].set_title('Jain')

    for point in pathbased_df:
        color = int_to_color[point[2]]
        axs[2, 1].scatter(point[0], point[1], c=color)

    axs[2, 1].set_title('Pathbased')


    plt.show()

    return;

kmeans_clusters()


