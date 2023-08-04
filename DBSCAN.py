from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import rand_score
from sklearn.preprocessing import StandardScaler
from RandIndex import rand_index
from sklearn.neighbors import NearestNeighbors





# Reading data
train_data = pd.read_csv("temp.csv")
#------------------------------------------------------------

# Fetching Values
y = train_data[['class']].values
main_lables = np.reshape(y, (410, ))
train_data.drop("class", axis = 1, inplace = True)

data_x =  train_data.values

#------------------------------------------------------------

sc = StandardScaler()
x = sc.fit_transform(data_x)



neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(x)
distances, indices = nbrs.kneighbors(x)

distances = np.sort(distances, axis = 0) 
print(distances)
distances = distances[:, 1]
plt.rcParams['figure.figsize'] = (5,3)
plt.plot(distances)
plt.show()

# the maximum value at the curvature of the graph is eps
# min_samples is 2 becase each picture is one sample

my_dbscan = DBSCAN(eps=57, min_samples=2).fit(x)

prediction = my_dbscan.fit_predict(x)

rd = rand_score(main_lables, prediction)

print("Main Rand Score : ", rd)

print(prediction)
print("1 #########################################################################")
print(my_dbscan.labels_)
print("2 #########################################################################")
print(main_lables)

ri = rand_index(main_lables, prediction)

print("My Rand Index : ", ri)


