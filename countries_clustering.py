import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import mixture
from scipy.cluster.hierarchy import fcluster, ward
from sklearn.preprocessing import StandardScaler
from scipy.cluster import hierarchy
from scipy import stats
from sklearn import decomposition
from scipy import linalg



df = pd.read_csv("./Donnees_projet_2021/data.csv")

# save countries name
countries = df['country']

# and remove column
del df['country']
columns = df.columns

#Return the shape of the dataset
def get_dataset_sise():
    return df.shape

# SHow the number of NA values in the dataset
def get_nb_na_in_df():
    return df.isnull().sum().sum()

#fill the na values with the mean
def fill_na_values():
    x = df.fillna(df.mean())
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
def correlation(dataset, column_name):
    correlations = dataset.corr()
    print(correlations)
    print('')
    print(correlations[column_name])
    pd.plotting.scatter_matrix(dataset, alpha=0.5, diagonal='kde')

    plt.show()
    return;




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
    GDP = dataset['GDP']
    income = dataset['income']

    print(GDP[get_index('Australia')])
    print('')
    print('PIB Australie = ', predict_PIB(income[get_index('Australia')], slope, intercept))
    print('PIB United Kingdom = ', predict_PIB(income[get_index('United Kingdom')], slope, intercept))
    print('PIB United States = ', predict_PIB(income[get_index('United States')], slope, intercept))

    # use these values:
    print('value = ', )
    dataset['GDP'][get_index('Australia')] = predict_PIB(income[get_index('Australia')], slope, intercept)
    dataset['GDP'][get_index('United Kingdom')] = predict_PIB(income[get_index('United Kingdom')], slope, intercept)
    dataset['GDP'][get_index('United States')] = predict_PIB(income[get_index('United States')], slope, intercept)

    dataset['GDP'][get_index('Italy')] = predict_PIB(income[get_index('Italy')], slope, intercept)
    dataset['GDP'][get_index('Norway')] = predict_PIB(income[get_index('Norway')], slope, intercept)

    print("PIB Italie = ", predict_PIB(income[get_index('Italy')], slope, intercept))
    print('PIB Norvège = ', predict_PIB(income[get_index('Norway')], slope, intercept))

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
    print('valeur propres = ', valeur_propres)
    print('')
    print('vecteur propres : ', vecteur_propres)


    return (valeur_propres, vecteur_propres, pca.transform(Z))

def quality_representation(valeur_propres, i):
    cumsum = np.cumsum(valeur_propres)

    qualite_representation = cumsum[i] / cumsum[-1]
    print('Qualité de la représentation si on garde %d axes = %f' % (i + 1, qualite_representation))
    print('')

def part_inertie_axe_i(i, vp):
    vp_i = vp[i]
    s_vp = 0
    for k in range(len(vp)):
        if(k != i):
            s_vp += vp[k]
    return (vp_i/s_vp)

def graphe_valeurs_propres(valeurs_propres):
    x = np.arange(1, len(valeurs_propres) + 1)
    plt.plot(x, valeurs_propres)
    plt.title("Valeurs propres en fonction de l'indice")
    plt.show()

def graphe_inertie_cumulée(valeurs_propres, show_graph):
    x = np.arange(1, len(valeurs_propres) + 1)
    cumsum = np.cumsum(valeurs_propres)
    Y = []
    for i in range(len(valeurs_propres)):
        Y.append(cumsum[i] / cumsum[-1])

    if(show_graph):
        plt.plot(x, Y)
        plt.title('Inertie cumulée')

        plt.show()
    return Y

def projection_premier_plan(datas):
    for point in datas:
        plt.scatter(point[0], point[1], c='blue')
    plt.title('Projection dans le premier plan principal')
    plt.show()

def projection_second_plan(datas):
    for point in datas:
        plt.scatter(point[2], point[3], c='blue')
    plt.title('Projection dans le second plan principal')
    plt.show()


dataset = fill_na_values()
L = [('inflation', [104]), ('GDP', [1000000]), ('life_expectation', [0, 32.1]), ('income', [80600,75200])]
replace_outliers_by_mean(L, dataset)
Z = scale_dataset(dataset)
#correlation(dataset, 'child_mortality')
(slope_gdp, intercept_gdp, err_gdp) = linear_regression_income_GDP(dataset, False)
(slope_fertility, intercept_fertility, err_fertility) = linear_regression_mortality_fertility(dataset, False)

#the dataset with correct values
ds = approximate_outliers_and_na(dataset, slope_gdp, intercept_gdp)
Z = scale_dataset(ds)

(valeurs_propres, vecteurs_propres, transformed) = acp(Z)
#graphe_valeurs_propres(valeurs_propres)
print(graphe_inertie_cumulée(valeurs_propres, False))
quality_representation(valeurs_propres, 3)
projection_premier_plan(transformed)
projection_second_plan(transformed)



