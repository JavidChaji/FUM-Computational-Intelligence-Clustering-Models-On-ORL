import csv
import cv2
import pandas as pd
from math import ceil
from sklearn.cluster import KMeans

#Creating CSV Work File
df = pd.DataFrame()
rows = []

temp_ar = []
for i in range(0, 5600):
    temp_ar.append("f%s"%(i+1))
temp_ar.append("class") 

rows.append(temp_ar)

for k in range(1, 411): 
    image_features = []
    pic_lable = ceil((k/10))
    image = cv2.imread("ORL/" + str(k) + "_" + "%s.jpg"%(pic_lable), cv2.IMREAD_GRAYSCALE)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            image_features.append(image[i][j])
        image_features.append(pic_lable)
    rows.append(image_features)

with open("temp.csv", 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)