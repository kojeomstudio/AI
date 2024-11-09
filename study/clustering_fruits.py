import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# KMeans 에서 사용할 수 있게 2차원 배열 형태로 수정.
fruits = np.load('./datas/fruits.npy') # RGBA 4차원 원소
fruits_2d = fruits.reshape(-1, 128 * 128) # 128 x 128 size image
#print(friuits)

# 군집에 필요한 k 값 찾기. ( 엘보 ) ==================
inertia = []
for k_value in range(2, 7):
    km = KMeans(n_clusters=k_value, n_init='auto', random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_)

plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
#=============================================

# elbow 방법으로 찾은 값은 4로 나옴. (과일 데이터 기준))
km = KMeans(n_clusters=4, random_state=42)
km.fit(fruits_2d)

print(f"labels info : {np.unique(km.labels_, return_counts=True)}")
