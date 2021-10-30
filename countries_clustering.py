import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn import mixture
from scipy.cluster.hierarchy import fcluster, ward
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from scipy import stats
from sklearn import decomposition
from mlxtend.plotting import plot_pca_correlation_graph
from sklearn.metrics import silhouette_score


int_to_color = {-1: "black", 0: "brown",1: 'blue', 2:'red', 3:'green', 4: 'yellow', 5: 'orange', 6: 'purple', 7: 'grey', 8: "pink", 9: ""}

dataframe = pd.read_csv("./Donnees_projet_2021/data.csv")

# save countries name
countries = dataframe['country']

# and remove column
df = dataframe
del df['country']
columns = df.columns

#Return the shape of the dataset
def get_dataset_sise():
    return df.shape

# Show the number of NA values in the dataset
def get_nb_na_in_df():
    return df.isnull().sum().sum()

#fill the na values with the mean
def fill_na_values(ds):
    x = ds.fillna(ds.mean())
    return x

# Print the outlisers value (above or under the threshold_value
def get_outliers(column_name, threshold_value, comparison_type):
    if(comparison_type == 'upper'):
        i = 0
        for value in df[column_name]:
            if value > threshold_value:
                print("%s of %f for %s" % (column_name, value, countries[i]))
            i += 1
    else:
        i = 0
        for value in df[column_name]:
            if value < threshold_value:
                print("%s of %f for %s" % (column_name, value, countries[i]))
            i += 1
    return;

# return mean of a column
def get_mean(column_name):
    return df[column_name].mean()

#L is a list of tuple: (column_name, [values_to_replace])
def replace_outliers_by_mean(L, dataset):
    for t in L:
        column_name, values_to_replace = t[0],t[1]
        mean = dataset[column_name].mean()
        if(column_name == 'GDP'):
            mean = int(mean)
        dataset[column_name].replace(values_to_replace, mean, inplace=True)
    return dataset;


# Scaled the dataset
def scale_dataset(dataset):
    scaler = StandardScaler()
    scaler.fit(dataset)
    Z = scaler.transform(dataset)
    return Z

# Print Correlation matrix and show the scatter matrix
def correlation(dataset, column_name, show_graph):
    correlations = dataset.corr()
    print(correlations)
    print('')
    print(correlations[column_name])
    if(show_graph):
        pd.plotting.scatter_matrix(dataset, alpha=0.5, diagonal='kde')
        plt.show()
    return correlations;

def linear_regression_income_GDP(dataset, show_graph):
    X = dataset['income']
    Y = dataset['GDP']

    if(show_graph):
        for k in range(len(X)):
            plt.scatter(X[k], Y[k], c = 'red')
            plt.xlabel("Revenu net moyen par personne")
            plt.ylabel('PIB par habitant')

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    print('')
    print('Droite régression linéaire : Y = %f X + (%f)' %(slope, intercept))
    print('')
    print('Erreur approximation PIB = ', std_err)
    print('')

    if(show_graph):

        fitLine = predict_PIB(X, slope, intercept)
        plt.plot(X, fitLine, c='blue')
        plt.show()

    return (slope, intercept, std_err)

def predict_PIB(x, slope, intercept):
   return slope * x + intercept

def get_index(country):
    for i,e in enumerate(countries):
        if e == country:
            return i

#Approximate the outlisers values and the NA with the linear regression (GDP - income)
def approximate_outliers_and_na(dataset, slope, intercept):
    income = dataset['income']
    for count in ['Australia','United Kingdom','United States', 'Italy', 'Norway'  ]:
        dataset['GDP'][get_index(count)] = predict_PIB(income[get_index(count)], slope, intercept)
    return dataset;

def linear_regression_mortality_fertility(dataset, show_graph):
    X = dataset['child_mortality']
    Y = dataset['total_fertility']

    if(show_graph):
        for k in range(len(X)):
            plt.scatter(X[k], Y[k], c='red')
            plt.xlabel("Taux de mortalité")
            plt.ylabel('Infertilité')

    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    print('')
    print('Droite régression linéaire : Y = %f X + (%f)' %(slope, intercept))
    print('')
    print('Erreur approximation PIB = ', std_err)
    print('')

    if(show_graph):
        fitLine = predict_fertility(X, slope, intercept)
        plt.plot(X, fitLine, c='blue')
        plt.show()
    return (slope, intercept, std_err)

def predict_fertility(x, slope, intercept):
   return slope * x + intercept

#ACP
def acp(Z):
    pca = decomposition.PCA()
    pca.fit(Z)
    valeur_propres = pca.explained_variance_
    vecteur_propres = pca.components_
    return (valeur_propres, vecteur_propres, pca.transform(Z))

def quality_representation(valeur_propres, i):
    cumsum = np.cumsum(valeur_propres)
    qualite_representation = cumsum[i] / cumsum[-1]
    print('Qualité de la représentation si on garde %d axes = %f' % (i + 1, qualite_representation))

def graph_eighenvalues(eighenvalues):
    x = np.arange(1, len(eighenvalues) + 1)
    plt.plot(x, eighenvalues)
    plt.title("Valeurs propres en fonction de l'indice")
    plt.show()

#draw the graph of cumlated inertie
def graph_inertie_cumulated(eighenvalues, show_graph):
    x = np.arange(1, len(eighenvalues) + 1)
    cumsum = np.cumsum(eighenvalues)
    Y = []
    for i in range(len(eighenvalues)):
        Y.append(cumsum[i] / cumsum[-1])

    if(show_graph):
        plt.plot(x, Y)
        plt.title('Inertie cumulée')
        plt.show()
    return Y

#draw the projection in the first plan
def projection_first_plan(datas, plan):
    for point in datas:
        if(plan == 'first'):
            plt.scatter(point[0], point[1], c='blue')
    plt.title('Projection dans le premier plan principal')
    plt.show()

# draw the correlation circle in the first plan and return the correlation matrix
def draw_correlation_circle(Z):
    figure, correlation_matrix = plot_pca_correlation_graph(Z,
                                                            columns,
                                                            dimensions=(1, 2),
                                                            figure_axis_size=10)
    plt.show()
    return correlation_matrix


#Clustering

#CAH
def CAH(Z, t, ds, draw_graph, draw_cluster):
    print('')
    print('*** CAH ***')
    linked = hierarchy.linkage(Z, 'ward', optimal_ordering=True)

    if(draw_graph):
        hierarchy.dendrogram(linked, color_threshold=t)

        plt.show()

    clusters = fcluster(linked, t=t, criterion='distance')
    if(draw_cluster):
        draw_clusters(clusters, Z, None, 'CAH', False)
    return clusters

def draw_clusters(labels, ds, centers, title, is_k_mean):
    i = 0
    for point in ds:
        color = int_to_color[labels[i] + 1]
        plt.scatter(point[0], point[1], c=color)
        i += 1

    if(is_k_mean):
        for center in centers:
            plt.scatter(center[0], center[1], c="black")

    plt.title(title)
    plt.show()

# For 4 clusters maximum => return cluster with coutries names
def list_countries_per_clusters(labels):
    C1, C2, C3, C4, i = [], [], [], [], 0
    for label in labels:
        if(label == 0):
            C1.append(countries[i])
        elif(label == 1):
            C2.append(countries[i])
        elif label == 2:
            C3.append((countries[i]))
        elif label == 3:
            C4.append((countries[i]))
        i += 1
    return (C1, C2, C3, C4)

# compute Kmean algorithm and draw clusters
def kmeans_clusters(Z, nb_cluster, show_clusters):
    km = KMeans(n_clusters=nb_cluster, random_state=42, n_init=100)
    km.fit_predict(Z)
    if(show_clusters):
        draw_clusters(km.labels_, Z, km.cluster_centers_, 'Kmeans clustering', True)
    return km.labels_

#compute Spectral clustering
def spectral_culstering(df, nb_cluster, show_clusters):
    print('')
    print('*** Spectral Clustering ***')
    print('')
    clustering = SpectralClustering(n_clusters=nb_cluster, assign_labels = "discretize").fit(df)

    if(show_clusters):
        draw_clusters(clustering.labels_, Z, None, 'Spectral clustering', False)


    return (clustering.labels_, clustering.affinity_matrix_)

def gaussian_mixture(data, nb_components, show_clusters):
    print('*** Gaussian Mixture ***')
    print('')
    gmm = mixture.GaussianMixture(n_components=nb_components, n_init=100, max_iter=300).fit(data)
    prediction = gmm.predict(data)

    if (show_clusters):
        draw_clusters(prediction, Z, None, 'Gaussian mixture', False)

    return prediction

def db_scan(data, epsilon, min_sample, show_clusters):
    print('')
    print('*** DBSCAN ***')
    clustering = DBSCAN(eps=epsilon, min_samples=min_sample).fit(data)

    if (show_clusters):
        draw_clusters(clustering.labels_, Z, None, 'DBSCAN', False)

    return clustering.labels_

def score_silhouette(algo):
    for K in [2, 3, 4, 5, 7, 9]:
        print('')
        print('For %d clusters ' % K)
        score = 0
        if(algo == 'KMEAN'):

            score = silhouette_score(Z, kmeans_clusters(Z, K, False), metric='euclidean')

        elif(algo == 'SPECTRAL'):
            score = silhouette_score(Z, spectral_culstering(Z, K, False)[0], metric='euclidean')

        print('Silhouetter Score: %.3f' % score)

# return the index of the min if it's not in index_list
def get_min(index_list, L):
    min = L[0]
    index_min = 0
    for k in range(len(L)):
        if k not in index_list :
            if(L[k] < min):
                min = L[k]
                index_min = k
    return (index_min)

# Functions call here

dataset = fill_na_values(df)
L = [('inflation', [104]), ('GDP', [1000000]), ('life_expectation', [0, 32.1]), ('income', [80600,75200])]
replace_outliers_by_mean(L, dataset)
Z = scale_dataset(dataset)
correlations_matrix = correlation(dataset, 'child_mortality', False)
(slope_gdp, intercept_gdp, err_gdp) = linear_regression_income_GDP(dataset, False)
(slope_fertility, intercept_fertility, err_fertility) = linear_regression_mortality_fertility(dataset, False)

#the dataset with correct values
ds = approximate_outliers_and_na(dataset, slope_gdp, intercept_gdp)
Z = scale_dataset(ds)

(eighenvalues, eighenvectors, transformed) = acp(Z)
#graph_eighenvalues(eighenvalues)
print('')
print(graph_inertie_cumulated(eighenvalues, False))
print('')
quality_representation(eighenvalues, 3)
#projection_first_plan(transformed, "first")
#draw_correlation_circle(Z)
#CAH(Z, 17, dataset, False, True)

"""
cah_clusters = list_countries_per_clusters(CAH(Z, 17, ds, False, False))
print("CAH clusters")
print('')
print('Cluster1 : ', cah_clusters[0])
print('Cluster2 : ', cah_clusters[1])
print('Cluster3 : ', cah_clusters[2])
print('')

print('')
print('Silhouette score')
print('')
print("Spectral : ", score_silhouette("SPECTRAL"))


km_clusters = list_countries_per_clusters(kmeans_clusters(Z, 2, True))
print("Kmeans clusters")
print('')
print('Cluster1 : ', km_clusters[0])
print('Cluster2 : ', km_clusters[1])
print('Cluster3 : ', km_clusters[2])


spectral_clusters = list_countries_per_clusters(spectral_culstering(Z, 2, True)[0])
print("Spectral clusters")
print('')
print('Cluster1 : ', spectral_clusters[0])
print('Cluster2 : ', spectral_clusters[1])

print('')

#print("ARI entre Kmeans et CAH : ", adjusted_rand_score(CAH(Z, 17, ds, False, False), kmeans_clusters(Z, 3, False)))

#Gaussian mixture

gaussian_clusters = list_countries_per_clusters(gaussian_mixture(Z, 4, True))
print("Gaussian clusters")
print('')
print('Cluster1 : ', gaussian_clusters[0])
print('Cluster2 : ', gaussian_clusters[1])
print('Cluster3 : ', gaussian_clusters[2])
print('Cluster4 : ', gaussian_clusters[3])
print('')
#print("ARI entre Gaussian et Spectral : ", adjusted_rand_score(gaussian_mixture(Z, 2, False), spectral_culstering(Z, 2, False)[0]))

#DBSCAN

#Find the correct parameters
for epsilon in [ 1.5, 1.75, 2, 3, 3.5, 4, 4.5, 5]:

    for min_sample in [3, 5, 7, 10, 12, 15]:
        score =  silhouette_score(Z, db_scan(Z, epsilon, min_sample, False), metric='euclidean')
        print("Score = %f for epsilon = %f and min_sample = %d" %(score, epsilon, min_sample))

dbs_clusters = list_countries_per_clusters(db_scan(Z, 1.5, 10, True))
print("DBSCAN clusters")
print('')
print('Cluster1 : ', dbs_clusters[0])
print('Cluster2 : ', dbs_clusters[1])
print('Cluster3 : ', dbs_clusters[2])
print('')
print("ARI entre Gaussian et Spectral : ", adjusted_rand_score(gaussian_mixture(Z, 2, False), spectral_culstering(Z, 2, False)[0]))

"""

#Si on choisi le Spectral Clustering

algo = 'kmean'
cluster_poor_countries, cluster_rich_countries = [], []

if(algo == "Spectral"):
    sc = spectral_culstering(Z, 2, False)
    spectral_clusters = list_countries_per_clusters(sc[0])
    af_matrix = sc[1]
    cluster_poor_countries = spectral_clusters[1]
    cluster_rich_countries = spectral_clusters[0]

elif algo == "kmean":
    km = kmeans_clusters(Z, 2, False)
    kmean_clusters = list_countries_per_clusters(km)
    cluster_poor_countries = kmean_clusters[0]
    cluster_rich_countries = kmean_clusters[1]

copy_countries = countries.tolist()

new_dataframe = pd.read_csv("./Donnees_projet_2021/data.csv")
print(new_dataframe)
for country in cluster_rich_countries:
    lst = np.array(copy_countries)
    result = copy_countries.index(country)
    print(country)
    new_dataframe.drop([result], inplace=True)

print('New dataframe from poor countries clustering')
print(new_dataframe)
new_countries = new_dataframe['country']
del new_dataframe['country']
new_dataframe = fill_na_values(new_dataframe)
L = [('inflation', [104]), ('GDP', [1000000]), ('life_expectation', [0, 32.1]), ('income', [80600,75200])]
replace_outliers_by_mean(L, new_dataframe)
# => Environ 50 pays restants, on doit en choisir 10

#With mean score for each country
def with_mean_score():
    new_Z = scale_dataset(new_dataframe)
    row_means = new_Z.mean(axis=1)
    liste_index_countries_to_help = []

    for k in range(10):
        liste_index_countries_to_help.append(get_min(liste_index_countries_to_help, row_means))
    print(liste_index_countries_to_help)

    liste_countries_to_help = []
    np_countries = new_countries.to_numpy()
    for index in liste_index_countries_to_help:
        liste_countries_to_help.append(np_countries[index])

    print(liste_countries_to_help)
    print('')
    print(Z.mean(axis=1))

sorted_df_by_gdp = new_dataframe.sort_values(by=['GDP'])
sorted_df_by_child_mortality = new_dataframe.sort_values(by=['child_mortality'])

list_countries_with_lowest_gdp = []
list_countries_with_highest_cm = []

k = 0
while k < 20:
    list_countries_with_lowest_gdp.append(countries[sorted_df_by_gdp.index[k]])
    list_countries_with_highest_cm.append(countries[sorted_df_by_child_mortality.index[-1-k]])
    k += 1

print('Pays avec le plus faible PIB : ', list_countries_with_lowest_gdp)
print('Pays avec le plus haut taux de mortalité chee les enfants de moins de 5 ans: ', list_countries_with_highest_cm)


final_countries = []
for c in list_countries_with_highest_cm:
    if c in list_countries_with_lowest_gdp:
        final_countries.append(c)

print('')
print('Liste des pays à aider en priorité: ')
print(final_countries[:10])


