import pandas as pd
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#讀取檔案
with open('mdata_5char.txt', 'r',encoding="utf-8") as file:
    lines = file.readlines()

x_values = []
y_values = []
z_values = []

for line in lines:
    # 跳過以"第"開頭的行
    if line.startswith("第"):
        continue
    if line.startswith("0"):
        continue
    data = line.split()

    avg = round(float(data[0]) * 100)
    x = round(float(data[1]) * 100)
    y = round(float(data[2]) * 100)
    z = round(float(data[3]) * 100)
    d = round(float(data[4]) * 100)

    x_values.append(x)
    y_values.append(y)
    # z_values.append(z)

#print("陰陽:", x_values)
#print("虛實:", y_values)



df = pd.DataFrame({
    "虛實":x_values,
    "陰陽":y_values,
    # "左右":z_values
})
k=12

kmeans =cluster.KMeans(n_clusters=k,random_state=12)
kmeans.fit(df)
print(kmeans.labels_)
colmap = np.array(["b","g","y","r","c","k","m","pink","orange","gray","brown","indigo"])
plt.scatter(df["虛實"],df["陰陽"],color=colmap[kmeans.labels_])
plt.show()