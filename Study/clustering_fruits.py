import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

friuits = np.load('./Datas/fruits.npy')
friuits_2d = friuits.reshape(-1, 128 * 128) # 128 x 128 size image
#print(friuits)

print(f"labels info : {np.unique(km.labels_, return_counts=True)}")

# 군집에 필요한 k 값 찾기. ( 엘보 ) ==================
inertia = []
for k_value in range(2, 7):
    km = KMeans(n_clusters=k_value, n_init='auto', random_state=42)
    km.fit(friuits_2d)
    inertia.append(km.inertia_)

plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
#=============================================

# elbow 방법으로 찾은 값은 4로 나옴. (과일 데이터 기준))
km = KMeans(n_clusters=4, random_state=42)
km.fit(friuits_2d)