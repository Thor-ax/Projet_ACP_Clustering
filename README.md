# Projet_ACP_Clustering
Analyse des principaux pays du monde en matière de développement. Aide à la décision pour les organisations de solidarité internationale. 

Deux fichier python: 

1) Le fichier main.py 

Il contient les tests des 5 méthodes (DBSCAN, Spectral clustering, Kmeans, CAH et Gaussian mixture) sur des datasets pour lesquels on connait déjà le clustering optimal.

Lancer kmeans_clusters() pour effectuer un clustering avec le k-means et afficher ce partitionnement.
Lancer spectral_clusters("kmeans") pour effectuer un clustering avec le Spectral clustering (le k-means comme méthode d'assignation) et afficher ce partitionnement.
Lancer gaussian_clusters() pour effectuer un clustering avec le gaussian mixture et afficher ce partitionnement.
Lancer db_scan_clusters(epsilon, min_samples, ("aggregation" or "jain" or "pathbased")) pour effectuer un clustering avec le DBSCAN et afficher ce partitionnement.
Lancer cah_cluster() pour effectuer un clustering avec le CAH et afficher ce partitionnement.

2) Me fichier countries_clustering

Il applique ces méthodes à un nouveau dataset contenant des données sur 167 pays (données économiques et de santé). 
Tout d'abord, on observe le nuage de points et la distribution des données. 
Puis, on remplace les valeurs manquantes et les valeurs aberrantes (soit par la valeur moyenne, soit à l'aide d'une régression linéaire après étude des corrélations). 
On effectue donc une étude des corrélations.
Enfin, on applique une ACP aux données centrées réduites (réduction des dimensions) et on observe le nuage de points projeté dans les plans principaux. 
On utilise le cercle des corrélations pour valider les corrélations et observer la contribution des valeurs aux différents axes.
Suite à cela, on applique les méthodes de clustering et on les compare afin de trouver la plus adapté à notre étude.
Enfin, on utilise la ou les méthodes choisie(s) afin de répondre à la question initiale et proposer une liste de pays à aider en priorité.
