#Jason Yeoh
#CS584 HW2

#Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy import stats
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy.linalg as linalg
import sklearn.cluster as cluster
import sklearn.neighbors as neighbors
import math
from kmodes.kmodes import KModes
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Load datasets
cars = pd.read_csv('cars.csv')
groceries = pd.read_csv('Groceries.csv')
four_circle = pd.read_csv('FourCircle.csv')

# (QUESTION 1)
# Calculate the frequency table of number of items purchase
nItemPurchase = groceries.groupby('Customer').size().sort_values()
freqTable = pd.Series.sort_index(pd.Series.value_counts(nItemPurchase))
print('Frequency of Number of Items Purchase')
print(freqTable)

sns.distplot(nItemPurchase, kde=False, hist=True).set_title('Number of Unique Items')
print('25th Percentile: %.2f'% np.percentile(nItemPurchase, 25))
print('50th Percentile: %.2f'% np.percentile(nItemPurchase, 50))
print('75th Percentile: %.2f'% np.percentile(nItemPurchase, 75))

#(1B)
ListItem = groceries.groupby(['Customer'])['Item'].apply(list).values.tolist()
numList = len(ListItem)

# Convert the Item List format to the Item Indicator format
te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
ItemIndicator = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(ItemIndicator, min_support = 75/numList, use_colnames = True)
frequent_itemsets['k_itemset'] = frequent_itemsets.apply(lambda x: len(x['itemsets']), axis=1)
frequent_itemsets.sort_values('k_itemset')


#(1C)
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print('There are {} association rules with confidence metric'.format(len(assoc_rules)))

#(1D)
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

#(1E)
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.6)
assoc_rules


# QUESTION 2
#(2A)
trainData = cars[['Type', 'Origin', 'DriveTrain', 'Cylinders']]
trainData['Type'].value_counts()

#(2B)
trainData['DriveTrain'].value_counts()

#(2C)
def compare(data1, data2, f1, f2):
    dist = 0
    
    for i, first in data1.iterrows():
        for j, second in data2.iterrows():
            if first['Type'] != second['Type']:
                dist += 1/f1 + 1/f2
            if first['Origin'] != second['Origin']:
                dist += 1/f1 + 1/f2
            if first['DriveTrain'] != second['DriveTrain']:
                dist += 1/f1 + 1/f2
            if first['Cylinders'] != second['Cylinders']:
                dist += 1/f1 + 1/f2
    return dist

#Treat missing values with 0.0
trainData["Cylinders"].fillna(0.0, inplace = True) 
compare(trainData[ trainData['Cylinders'] == 5.0], trainData[ trainData['Cylinders'] == 0.0], 7, 2)
compare(trainData[trainData['Origin']=='Asia'], trainData[trainData['Origin']=='Europe'], 158, 123)

#(2D)
def k_modes(data, clusters):
    km = KModes(n_clusters=clusters, init='Huang', n_init=5, verbose=1)
    clusters = km.fit_predict(data)
    centroids = km.cluster_centroids_
    
    return clusters, centroids

clusters, centroids = k_modes(trainData, 3)
trainData['clusters'] = clusters
trainData[trainData['clusters'] == 0]['Origin'].value_counts()
trainData[trainData['clusters'] == 1]['Origin'].value_counts()
trainData[trainData['clusters'] == 2]['Origin'].value_counts()


# QUESTION 3
#(3A) Under observation, there are FOUR clusters
sns.scatterplot(x='x', y='y', data=four_circle)

#(3B)
train = four_circle[['x','y']]
kmeans = cluster.KMeans(n_clusters=4, random_state=60616).fit(train)

print("Cluster Centroids = \n", kmeans.cluster_centers_)

four_circle['KMeanCluster'] = kmeans.labels_

plt.scatter(four_circle['x'], four_circle['y'], c = four_circle['KMeanCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

#(3C)
kNNSpec = neighbors.NearestNeighbors(n_neighbors = 10, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(train)
d3, i3 = nbrs.kneighbors(train)

# Retrieve the distances among the observations
distObject = neighbors.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(train)

nObs = train.shape[0]

# Create the Adjacency matrix
Adjacency = np.zeros((nObs, nObs))
for i in range(nObs):
    for j in i3[i]:
        Adjacency[i,j] = math.exp(- (distances[i][j])**2 )

# Make the Adjacency matrix symmetric
Adjacency = 0.5 * (Adjacency + Adjacency.transpose())

# Create the Degree matrix
Degree = np.zeros((nObs, nObs))
for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum

#(3D)
# Create the Laplacian matrix        
Lmatrix = Degree - Adjacency

# Obtain the eigenvalues and the eigenvectors of the Laplacian matrix
evals, evecs = linalg.eigh(Lmatrix)

# Series plot of the smallest fifteen eigenvalues to determine the number of neighbors
sequence = np.arange(1,16,1) 
plt.plot(sequence, evals[0:15,], marker = "o")
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.grid("both")
plt.xticks(sequence)
plt.show()


#(3E)
Z = evecs[:,[0,1]]

kmeans_spectral = cluster.KMeans(n_clusters = 4, random_state = 60616).fit(Z)
train['SpectralCluster'] = kmeans_spectral.labels_

plt.scatter(train['x'], train['y'], c = train['SpectralCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()