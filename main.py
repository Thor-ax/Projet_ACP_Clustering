import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import mixture
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy

aggregation_df = np.loadtxt("./Donnees_projet_2021/aggregation.txt")
jain_df = np.loadtxt("./Donnees_projet_2021/jain.txt")
pathbased_df = np.loadtxt("./Donnees_projet_2021/pathbased.txt")

datas = [(aggregation_df, [0, 1], 'Aggregation'),(jain_df, [1, 1], 'jain'), (pathbased_df, [2, 1], 'Pathbased') ]
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

# compute Kmean algorithm and draw clusters
def kmeans_clusters():
    fig, axs = plt.subplots(3, 2)
    draw_k_means(axs)
    draw_points(axs)

    plt.show()

# Draw the scatter points from the dataset (aggregation, jain, pathbased)
def draw_points(axs):
    for data in datas:
        for point in data[0]:
            color = int_to_color[point[2]]
            axs[data[1][0], data[1][1]].scatter(point[0], point[1], c=color)

        axs[data[1][0], data[1][1]].set_title(data[2])

#compute K-means algorithm and return the prediction
def k_means(dataset, K):
    km = KMeans(n_clusters=K, random_state=42, n_init=100)
    predict = km.fit_predict(dataset)
    return (km, predict)

#draw the Kmean clustering
def draw_k_means(axs):
    L = [(aggregation_df, 7, 'Aggregation with Kmeans', 0), (jain_df, 2, 'jain with Kmeans', 1), (pathbased_df, 3, 'pathbased with Kmeans', 2)]
    for X in L:
        (km, labels) = k_means(X[0], X[1])
        i = 0
        for point in X[0]:
            color = int_to_color[labels[i] + 1]
            axs[X[3], 0].scatter(point[0], point[1], c=color)
            i += 1

        for center in km.cluster_centers_:
            axs[X[3], 0].scatter(center[0], center[1], c="black")

        axs[X[3], 0].set_title(X[2])

#Cmpute Gaussian Mixture and return the predicted labels
def gaussian_mixture(data, nb_components):
    print('*** Gaussian Mixture ***')
    print('')
    gmm = mixture.GaussianMixture(n_components=nb_components, n_init=100, max_iter=300).fit(data)
    """
    print('Mean: ')
    print('')
    print(gmm.means_)
    print('')
    print('Covariance: ')
    print('')
    print(np.sqrt(gmm.covariances_))
    """
    prediction = gmm.predict(data)

    return prediction

def gaussian_clusters():
    fig, axs = plt.subplots(3, 2)
    draw_gaussian_clusters(axs)
    draw_points(axs)
    plt.show()

#Dras the Gaussian clustering
def draw_gaussian_clusters(axs):
    L = [(aggregation_df, 7, 'Aggregation with Gaussian mixture', 0), (jain_df, 2, 'jain with Gaussian mixture', 1),
         (pathbased_df, 3, 'pathbased with Gaussian mixture', 2)]
    for X in L:
        labels = gaussian_mixture(X[0], X[1])
        i = 0
        for point in X[0]:
            color = int_to_color[labels[i] + 1]
            axs[X[3], 0].scatter(point[0], point[1], c=color)
            i += 1

        axs[X[3], 0].set_title(X[2])

#Compute the DBSCAN and return the labels
def db_scan(data, epsilon, min_sample):
    print('')
    print('*** DBSCAN ***')
    clustering = DBSCAN(eps=epsilon, min_samples=min_sample).fit(data)

    return clustering.labels_

def db_scan_clusters(epsilon, min_sample, data):
    fig, axs = plt.subplots(2, 1)
    draw_dbscan_clusters(axs, epsilon, min_sample, data)
    draw_scatter_points(axs, data)

    plt.show()

#Draw the scatter points of one dataset (aggregation or jain or pathbased)
def draw_scatter_points(axs, data):
    datas = {"aggregation": (aggregation_df, 'Aggregation'), "jain": (jain_df, 'Jain')
        , "pathbased": (pathbased_df, 'Pathbased')}

    for point in datas[data][0]:
        color = int_to_color[point[2]]
        axs[0].scatter(point[0], point[1], c=color)

    axs[0].set_title(datas[data][1])

#Draw the DBSCAN clustering
def draw_dbscan_clusters(axs, epsilon, min_sample, data):
    datas = {"aggregation" : (aggregation_df, 'Aggregation with DBSCAN'), "jain": (jain_df, 'Jain with DBSCAN')
        , "pathbased": (pathbased_df, 'Pathbased with DBSCAN') }

    labels = db_scan(datas[data][0], epsilon, min_sample)
    i = 0
    for point in datas[data][0]:
        color = int_to_color[labels[i]]
        axs[1].scatter(point[0], point[1], c=color)
        i += 1

    axs[1].set_title(datas[data][1])

#Run DBSCAN with different parameters to find the best
def get_dbscan_parameters(df):
    eps, ms, ari = 0.5, 3, -1
    for epsilon in [0.5, 1, 1.5, 2, 2.5, 3, 4, 5]:
        for min_samples in range(3, 100):
            labels1, labels2 = db_scan(data=df, epsilon = epsilon, min_sample = min_samples), df[:, [-1]]
            labels2 = np.transpose(labels2)[0]
            rd_score = adjusted_rand_score(labels1, labels2)
            if(rd_score > ari):
                ari = rd_score
                eps = epsilon
                ms = min_samples
    return (epsilon, ms, ari)

#Compute the spectral clustering and return the labels
def spectral_culstering(df, nb_cluster, assign_method):
    print('')
    print('*** Spectral Clustering ***')
    print('')
    clustering = SpectralClustering(n_clusters=nb_cluster, assign_labels = assign_method).fit(df)

    return clustering.labels_

def spectral_clusters(assign_method):
    fig, axs = plt.subplots(3, 2)
    draw_spectral_cluster(axs, assign_method)
    draw_points(axs)
    plt.show()

#plot the Spectral clustering
def draw_spectral_cluster(axs, assign_method):
    L = [(aggregation_df, 7, 'Aggregation with Spectral clustering', 0), (jain_df, 2, 'jain with Spectral clustering', 1),
         (pathbased_df, 3, 'pathbased with Spectral clustering', 2)]
    for X in L:
        labels = spectral_culstering(X[0], X[1], assign_method)
        i = 0
        for point in X[0]:
            color = int_to_color[labels[i] + 1]
            axs[X[3], 0].scatter(point[0], point[1], c=color)
            i += 1

        axs[X[3], 0].set_title(X[2])

#COmpute the CAH and return the labels
def CAH(df, t):
    print('')
    print('*** CAH ***')
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    Z = hierarchy.linkage(scaled, 'ward', optimal_ordering=True)

    #dn = hierarchy.dendrogram(Z, color_threshold=t)
    #plt.title('Aggregation dendrogramme for t = 8')
    #plt.show()

    clusters = fcluster(Z, t=t, criterion='distance')
    return clusters

def cah_cluster():
    fig, axs = plt.subplots(3, 2)
    draw_cah_cluster(axs)
    draw_points(axs)
    plt.show()

#Plot the CAH clustering
def draw_cah_cluster(axs):
    L = [(aggregation_df, 8, 'Aggregation with CAH', 0),
         (jain_df, 20, 'jain with CAH', 1),
         (pathbased_df, 19, 'pathbased with CAH', 2)]
    for X in L:
        labels = CAH(X[0], X[1])
        i = 0
        for point in X[0]:
            color = int_to_color[labels[i] + 1]
            axs[X[3], 0].scatter(point[0], point[1], c=color)
            i += 1

        axs[X[3], 0].set_title(X[2])

#Return adjusted rand score from given labels and dataset
def rand_scores(labels1, algo, dataset_name, df):
    print(algo)
    print('')
    labels2 = df[:, [-1]]
    labels2 = np.transpose(labels2)[0]
    print("For %s, ARI = %f " % (dataset_name, adjusted_rand_score(labels2, labels1)))

#kmeans_clusters()
#cah_cluster()
# aggregation => t = 8
# jain => t = 20
# pathbased => t = 19
#CAH(aggregation_df, 8)

#get_dbscan_parameters(pathbased_df)

# epsilon = 1.5 et min_samples = 3 pour aggregation => ARI = 1
# epsilon = 2.5 et min_samples = 15 pour Jain => ARI = 1
#epsilon = 2 et min_samples = 3 pour pathbased => ARI = 0.9858
#db_scan_clusters(2, 3, "pathbased")

#spectral_clusters("kmeans")
#gaussian_clusters()

#ARI

#Kmeans
rand_scores(k_means(aggregation_df, 7)[1], "Kmeans", 'Aggregation', aggregation_df)
rand_scores(k_means(jain_df, 2)[1], "Kmeans", 'Jain', jain_df)
rand_scores(k_means(pathbased_df, 3)[1], "Kmeans", 'Pathbased', pathbased_df)

#Spectral clustering
rand_scores(spectral_culstering(aggregation_df, 7, "discretize"), "Spectral clustering", 'Aggregation', aggregation_df)
rand_scores(spectral_culstering(jain_df, 2, "discretize"), "Spectral clustering", 'Jain', jain_df)
rand_scores(spectral_culstering(pathbased_df, 3, "discretize"), "Spectral clustering", 'Pathbased', pathbased_df)

#Gaussian mixture
rand_scores(gaussian_mixture(aggregation_df, 7), "Gaussian mixture", 'Aggregation', aggregation_df)
rand_scores(gaussian_mixture(jain_df, 2), "Gaussian mixture", 'Jain', jain_df)
rand_scores(gaussian_mixture(pathbased_df, 3), "Gaussian mixture", 'Pathbased', pathbased_df)

#CAH
rand_scores(CAH(aggregation_df, 8), "CAH", 'Aggregation', aggregation_df)
rand_scores(CAH(jain_df, 20), "CAH", 'Jain', jain_df)
rand_scores(CAH(pathbased_df, 19), "CAH", 'Pathbased', pathbased_df)

#DBSCAN
rand_scores(db_scan(aggregation_df, 1.5, 3), "DBSCAN", 'Aggregation', aggregation_df)
rand_scores(db_scan(jain_df, 2.5, 15), "DBSCAN", 'Jain', jain_df)
rand_scores(db_scan(pathbased_df, 2, 3), "DBSCAN", 'Pathbased', pathbased_df)

