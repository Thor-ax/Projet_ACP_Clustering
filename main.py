import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import mixture
from scipy.cluster.hierarchy import fcluster, ward
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy



aggregation_df = np.loadtxt("./Donnees_projet_2021/aggregation.txt")
jain_df = np.loadtxt("./Donnees_projet_2021/jain.txt")
pathbased_df = np.loadtxt("./Donnees_projet_2021/pathbased.txt")

int_to_color = {-1: "black", 0: "brown",1: 'blue', 2:'red', 3:'green', 4: 'yellow', 5: 'orange', 6: 'purple', 7: 'grey', 8: "pink", 9: ""}

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

# compute Kmean algorithm and draw clusters
def kmeans_clusters():
    fig, axs = plt.subplots(3, 2)
    draw_k_means(axs)
    draw_points(axs)

    plt.show()

    return;

def draw_points(axs):
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
    return;

def draw_k_means(axs):

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
    return;

def k_means(dataset, K):
    km = KMeans(n_clusters=K)
    predict = km.fit_predict(dataset)
    return(km, predict)

"""

prediction = k_means(aggregation_df, 7)[1]
labels2 = aggregation_df[:, [-1]]
labels2 = np.transpose(labels2)[0]

print('Aggregation : ')
print('')
print('ARI = ', adjusted_rand_score(prediction, labels2))

print('')
prediction = k_means(jain_df, 2)[1]
labels2 = jain_df[:, [-1]]
labels2 = np.transpose(labels2)[0]

print('Jain : ')
print('')
print('ARI = ', adjusted_rand_score(prediction, labels2))

print('')

prediction = k_means(pathbased_df, 3)[1]
print(prediction)
print('')
print(prediction.shape)
labels2 = pathbased_df[:, [-1]]
labels2 = np.transpose(labels2)[0]

print('Pathbased : ')
print('')
print('ARI = ', adjusted_rand_score(labels2, prediction))

"""

def gaussian_mixture(data, nb_components):
    print('*** Gaussian Mixture ***')
    print('')
    gmm = mixture.GaussianMixture(n_components=nb_components, n_init=100, max_iter=300).fit(data)

    print('Mean: ')
    print('')
    print(gmm.means_)
    print('')
    print('Covariance: ')
    print('')
    print(np.sqrt(gmm.covariances_))
    print('')
    print('Prediction')
    print('')
    prediction = gmm.predict(data)

    return prediction


def gaussian_clusters():
    fig, axs = plt.subplots(3, 2)
    draw_gaussian_clusters(axs)
    draw_points(axs)

    plt.show()

    return;

def draw_gaussian_clusters(axs):
    # Aggregation
    gm_aggregation_labels = gaussian_mixture(aggregation_df, 7)

    i = 0
    for point in aggregation_df:
        color = int_to_color[gm_aggregation_labels[i] + 1]
        axs[0, 0].scatter(point[0], point[1], c=color)
        i += 1

    axs[0, 0].set_title('Aggregation with Gaussian Mixture')

    # Jain
    gm_jain_labels = gaussian_mixture(jain_df, 2)
    i = 0
    for point in jain_df:
        color = int_to_color[gm_jain_labels[i] + 1]
        axs[1, 0].scatter(point[0], point[1], c=color)
        i += 1

    axs[1, 0].set_title('Jain with Gaussian Mixture')

    # Pathbased
    gm_pathbased_labels = gaussian_mixture(pathbased_df, 3)
    i = 0
    for point in pathbased_df:
        color = int_to_color[gm_pathbased_labels[i] + 1]
        axs[2, 0].scatter(point[0], point[1], c=color)
        i += 1

    axs[2, 0].set_title('Pathbased with Gaussian Mixture')

    return;

"""

prediction = gaussian_mixture(aggregation_df, 7)
labels2 = aggregation_df[:, [-1]]
labels2 = np.transpose(labels2)[0]

print('Aggregation : ')
print('')
print('ARI = ', adjusted_rand_score(prediction, labels2))

print('')
prediction = gaussian_mixture(jain_df, 2)
labels2 = jain_df[:, [-1]]
labels2 = np.transpose(labels2)[0]

print('Jain : ')
print('')
print('ARI = ', adjusted_rand_score(prediction, labels2))

print('')

prediction = gaussian_mixture(pathbased_df, 3)
print(prediction)
print('')
print(prediction.shape)
labels2 = pathbased_df[:, [-1]]
labels2 = np.transpose(labels2)[0]

print('Pathbased : ')
print('')
print('ARI = ', adjusted_rand_score(labels2, prediction))

"""


def db_scan(data, epsilon, min_sample):
    print('')
    print('*** DBSCAN ***')
    clustering = DBSCAN(eps=epsilon, min_samples=min_sample).fit(data)
    print('Clustering: ')
    print(clustering)

    return clustering.labels_

def db_scan_clusters(epsilon, min_sample, data):
    fig, axs = plt.subplots(2, 1)
    draw_dbscan_clusters(axs, epsilon, min_sample, data)
    draw_scatter_points(axs, data)

    plt.show()

    return;

def draw_scatter_points(axs, data):

    if(data == "aggregation"):
        for point in aggregation_df:
            color = int_to_color[point[2]]
            axs[0].scatter(point[0], point[1], c=color)

        axs[0].set_title('Aggregation')

    if(data == "jain"):
        for point in jain_df:
            color = int_to_color[point[2]]
            axs[0].scatter(point[0], point[1], c=color)

        axs[0].set_title('Jain')

    if(data == "pathbased"):
        for point in pathbased_df:
            color = int_to_color[point[2]]
            axs[0].scatter(point[0], point[1], c=color)

        axs[0].set_title('Pathbased')

    return;


def draw_dbscan_clusters(axs, epsilon, min_sample, data):

    if (data == "aggregation"):
        # Aggregation
        ds_aggregation_labels = db_scan(aggregation_df, epsilon, min_sample)

        print(ds_aggregation_labels)

        i = 0
        for point in aggregation_df:
            color = int_to_color[ds_aggregation_labels[i]]
            axs[1].scatter(point[0], point[1], c=color)
            i += 1

        axs[1].set_title('Aggregation with DBSCAN')

    if(data == "jain"):
        # Jain
        ds_jain_labels = db_scan(jain_df, epsilon, min_sample)
        i = 0
        for point in jain_df:
            color = int_to_color[ds_jain_labels[i]]
            axs[1].scatter(point[0], point[1], c=color)
            i += 1

        axs[1].set_title('Jain with DBSCAN')

    if(data == "pathbased"):
        # Pathbased
        ds_pathbased_labels = db_scan(pathbased_df, epsilon, min_sample)
        i = 0
        for point in pathbased_df:
            color = int_to_color[ds_pathbased_labels[i]]
            axs[1].scatter(point[0], point[1], c=color)
            i += 1

        axs[1].set_title('Pathbased with DBSCAN')

    return;

def get_dbscan_parameters(df):
    eps = 0.5
    ms = 3
    ari = -1
    for epsilon in [0.5, 1, 1.5, 2, 2.5, 3, 4, 5]:

        for min_samples in range(3, 100):
            labels1 = db_scan(data=df, epsilon = epsilon, min_sample = min_samples)
            labels2 = df[:, [-1]]
            labels2 = np.transpose(labels2)[0]
            rd_score = adjusted_rand_score(labels1, labels2)
            if(rd_score > ari):

                ari = rd_score
                eps = epsilon
                ms = min_samples

    print('ARI = ', ari)
    print('epsilon = ', eps)
    print("min_samples = ", ms)
    return (epsilon, ms)


# epsilon = 1.5 et min_samples = 3 pour aggregation => ARI = 1
# epsilon = 2.5 et min_samples = 15 pour Jain => ARI = 1
#epsilon = 2 et min_samples = 3 pour pathbased => ARI = 0.9858

#get_dbscan_parameters(pathbased_df)

#db_scan_clusters(2, 3, "pathbased")


def spectral_culstering(df, nb_cluster, assign_method):
    print('')
    print('*** Spectral Clustering ***')
    print('')
    clustering = SpectralClustering(n_clusters=nb_cluster, assign_labels = assign_method).fit(df)
    print(clustering)

    return clustering.labels_

def spectral_clusters(assign_method):
    fig, axs = plt.subplots(3, 2)
    draw_spectral_cluster(axs, assign_method)
    draw_points(axs)

    plt.show()

    return;

def draw_spectral_cluster(axs, assign_method):
    # Aggregation
    sc_aggregation_labels = spectral_culstering(aggregation_df, 7, assign_method)

    i = 0
    for point in aggregation_df:
        color = int_to_color[sc_aggregation_labels[i] + 1]
        axs[0, 0].scatter(point[0], point[1], c=color)
        i += 1

    axs[0, 0].set_title('Aggregation with Spectral clustering')

    # Jain
    sc_jain_labels = spectral_culstering(jain_df, 2, assign_method)
    i = 0
    for point in jain_df:
        color = int_to_color[sc_jain_labels[i] + 1]
        axs[1, 0].scatter(point[0], point[1], c=color)
        i += 1

    axs[1, 0].set_title('Jain with Spectral clustering')

    # Pathbased
    sc_pathbased_labels = spectral_culstering(pathbased_df, 3, assign_method)
    i = 0
    for point in pathbased_df:
        color = int_to_color[sc_pathbased_labels[i] + 1]
        axs[2, 0].scatter(point[0], point[1], c=color)
        i += 1

    axs[2, 0].set_title('Pathbased with Spectral clustering')

    return;

#spectral_clusters("kmeans")

"""
prediction = spectral_culstering(aggregation_df, 7, "kmeans")
labels2 = aggregation_df[:, [-1]]
labels2 = np.transpose(labels2)[0]

print('Aggregation : ')
print('')
print('ARI = ', adjusted_rand_score(prediction, labels2))

print('')
prediction = spectral_culstering(jain_df, 2, "kmeans")
labels2 = jain_df[:, [-1]]
labels2 = np.transpose(labels2)[0]

print('Jain : ')
print('')
print('ARI = ', adjusted_rand_score(prediction, labels2))

print('')

prediction = spectral_culstering(pathbased_df, 3, "kmeans")
print(prediction)
print('')
print(prediction.shape)
labels2 = pathbased_df[:, [-1]]
labels2 = np.transpose(labels2)[0]

print('Pathbased : ')
print('')
print('ARI = ', adjusted_rand_score(labels2, prediction))

"""

def CAH(df, t):
    print('')
    print('*** CAH ***')
    scaler = StandardScaler()
    scaler.fit(df)
    scaled = scaler.transform(df)

    Z = hierarchy.linkage(scaled, 'ward', optimal_ordering=True)

    #dn = hierarchy.dendrogram(Z, color_threshold=t)

    #plt.show()

    clusters = fcluster(Z, t=t, criterion='distance')
    print('')
    print(clusters)
    return clusters

def cah_cluster():
    fig, axs = plt.subplots(3, 2)
    draw_cah_cluster(axs)
    draw_points(axs)

    plt.show()

    return;

def draw_cah_cluster(axs):
    # Aggregation
    cah_aggregation_labels = CAH(aggregation_df, 8)

    i = 0
    for point in aggregation_df:
        color = int_to_color[cah_aggregation_labels[i]]
        axs[0, 0].scatter(point[0], point[1], c=color)
        i += 1

    axs[0, 0].set_title('Aggregation with CAH')

    # Jain
    cah_jain_labels = CAH(jain_df, 20)
    i = 0
    for point in jain_df:
        color = int_to_color[cah_jain_labels[i]]
        axs[1, 0].scatter(point[0], point[1], c=color)
        i += 1

    axs[1, 0].set_title('Jain with Spectral clustering')

    # Pathbased
    cah_pathbased_labels = CAH(pathbased_df, 19)
    i = 0
    for point in pathbased_df:
        color = int_to_color[cah_pathbased_labels[i]]
        axs[2, 0].scatter(point[0], point[1], c=color)
        i += 1

    axs[2, 0].set_title('Pathbased with CAH')

    return;


cah_cluster()
# aggregation => t = 8
# jain => t = 20
# pathbased => t = 19