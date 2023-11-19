import pandas as pd
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

# 讀取文件
with open('mdata_5char.txt', 'r', encoding="utf-8") as file:
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
    z_values.append(z)

df = pd.DataFrame({
    "虛實": x_values,
    "陰陽": y_values,
    "左右": z_values
})

k = 12

kmeans = cluster.KMeans(n_clusters=k, random_state=12)
kmeans.fit(df)
print(kmeans.labels_)
colmap = np.array(["b", "g", "y", "r", "c", "k", "m", "pink", "orange", "gray", "brown", "indigo"])

# 創建一個三維散點圖
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 在三維圖中繪製資料點，使用k-means的標籤來區分不同的簇
ax.scatter(df["虛實"], df["陰陽"], df["左右"], c=colmap[kmeans.labels_])

# 設定坐標軸標籤
ax.set_xlabel('虛實')
ax.set_ylabel('陰陽')
ax.set_zlabel('左右')

plt.show()
