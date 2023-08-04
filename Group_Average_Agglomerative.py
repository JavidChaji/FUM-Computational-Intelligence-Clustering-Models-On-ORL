import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import rand_score
from sklearn.preprocessing import StandardScaler

from RandIndex import rand_index

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

# linkage Shows the Link Type

my_Agglomerative = AgglomerativeClustering(n_clusters=41, linkage='average').fit(x)

# prediction = my_Agglomerative.fit_predict(x, y)

prediction = my_Agglomerative.labels_

rd = rand_score(main_lables, prediction)

print(rd)


