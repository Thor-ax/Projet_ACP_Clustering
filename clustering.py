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


df= pd.read_csv("./Donnees_projet_2021/data.csv")

print(df.shape)

print('null ', df.isnull().sum().sum())



countries = df['country']

del df['country']

columns = df.columns

df = df.fillna(df.mean())

i = 0
for income in df['income']:
    if income > 50000:
        print("Income of %s = %d indice = %d" %(countries[i], income, i))
    i += 1

# valeurs aberrantes:
average_income = df['income'].mean()
df['income'].replace([80600,75200],average_income, inplace=True)

print('')

i = 0
for expectation in df['life_expectation']:
    if expectation < 50:
        print("Life expectation of %f for %s" %(expectation, countries[i]))
    i += 1

# valeurs aberrantes:
average_life_expectation = df['life_expectation'].mean()
df['life_expectation'].replace([0, 32.1],average_life_expectation, inplace=True)

print('')

i = 0
for inflation in df['inflation']:
    if inflation > 60:
        print("Inflation = %f for %s" %(inflation, countries[i]))
    i += 1

average_inflation = df['inflation'].mean()
df['inflation'].replace(104,average_inflation, inplace=True)

average_gdp = int(df['GDP'].mean())
df['GDP'].replace(1000000,average_gdp, inplace=True)

def hist():
    for column in columns:
        hist = df[column].hist()
        plt.title(column)
        plt.show()
    return;


def correlation():
    correlations = df.corr()
    print(correlations)
    print('')
    print(correlations['GDP'])
    pd.plotting.scatter_matrix(df, alpha=0.5, diagonal='kde')

    plt.show()
    return;


X = df['income']
Y = df['GDP']

for k in range(len(X)):
    plt.scatter(X[k], Y[k], c = 'red')
    plt.xlabel("Revenu net moyen par personne")
    plt.ylabel('PIB par habitant')


slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
def predict_PIB(x):
   return slope * x + intercept

print('Erreur approximation PIB = ', std_err)

fitLine = predict_PIB(X)
plt.plot(X, fitLine, c='blue')

def get_index(country):
    for i,e in enumerate(countries):
        if e == country:
            return i

# coefficients a et b de la regression linéaire Y = a x + b

print(get_index('Australia'))
GDP = df['GDP']
income = df['income']

print(GDP[get_index('Australia')])
print('')
print('PIB Australie = ', predict_PIB(income[get_index('Australia')]))
print('PIB United Kingdom = ', predict_PIB(income[get_index('United Kingdom')]))
print('PIB United States = ', predict_PIB(income[get_index('United States')]))

#use these values:
print('value = ', )
df['GDP'][get_index('Australia')] = predict_PIB(income[get_index('Australia')])
df['GDP'][get_index('United Kingdom')] = predict_PIB(income[get_index('United Kingdom')])
df['GDP'][get_index('United States')] = predict_PIB(income[get_index('United States')])

#Calculer la valeur manquante du PIB pour l'italie et la Norvège

df['GDP'][get_index('Italy')] = predict_PIB(income[get_index('Italy')])
df['GDP'][get_index('Norway')] = predict_PIB(income[get_index('Norway')])


print("PIB Italie = ", predict_PIB(income[get_index('Italy')]))
print('PIB Norvège = ', predict_PIB(income[get_index('Norway')]))

plt.show()


X = df['child_mortality']
Y = df['total_fertility']

for k in range(len(X)):
    plt.scatter(X[k], Y[k], c = 'red')
    plt.xlabel("Taux de mortalité")
    plt.ylabel('Infertilité')

slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
def predict_fertility(x):
   return slope * x + intercept

print(slope, intercept)
print('')

child_mortality = df['child_mortality']

print('Erreur approximation Total fertility = ', std_err)
print('Total fertility France = ', predict_fertility(child_mortality[get_index('France')]))
print('Total fertility Niger = ', predict_fertility(child_mortality[get_index('Niger')]))

df['total_fertility'][get_index('France')] = predict_fertility(child_mortality[get_index('France')])
df['total_fertility'][get_index('Niger')] = predict_fertility(child_mortality[get_index('Niger')])

fitLine = predict_fertility(X)
plt.plot(X, fitLine, c='blue')

plt.show()
