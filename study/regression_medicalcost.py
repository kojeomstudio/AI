import numpy as np
import pandas as pd

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # 표준화 (데이터 전처리))
import matplotlib.pyplot as plt # 그래프
from sklearn.linear_model import LinearRegression # 선형회귀
from sklearn.svm import SVR
from sklearn.linear_model import Lasso # 랏쏘회귀 ( 규제 )

import seaborn as sns

# CSV 파일 불러오기
file_path = './datas/MedicalCostDatasets.csv'
raw_data = pd.read_csv(file_path)

# One-Hot Encoding
# string 데이터 -> 정수형으로 변경 -> 값에 차이에 대한 컬럼 신규 생성.
raw_data = pd.get_dummies(raw_data, columns=['sex', 'smoker', 'region'])

# 데이터 확인
print("==== raw_data.head() ====")
print(f"{raw_data.head()}")

# 데이터 정보 확인
print(f"raw_data.info() : {raw_data.info()}")

# 기초 통계량 확인
print("raw_data.describe")
print(f"{raw_data.describe}")

print(f"columns : {raw_data.columns}")
print(f"raw_data shape : {raw_data.shape}")

x = raw_data.drop('charges', axis=1) # 비용을 제외한 나머지를 학습으로 아용.
y = raw_data['charges'] # (=정답)

print(f"x's shape : {x.shape}")
print(f"y's shape : {y.shape}")

# 데이터 나누기
# (훈련, 테스트, 검증)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 상관관계 행렬 및 히트맵
corr_matrix = raw_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
#plt.show()

# 표준화
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)  # 검증 데이터 표준화
x_test_scaled = scaler.transform(x_test)  # 테스트 데이터 표준화

# 학습 & 검증
lr = LinearRegression()

lr.fit(x_train_scaled, y_train)
print(f"score (by train data not scaled) : {lr.score(x_train, y_train)}") # 특성간에 값이 범위가 다르므로, 정규화를 꼭 해야함.
print(f" score (by scaled train data) :  {lr.score(x_train_scaled, y_train)}")
print(f" score (by scaled test data) :  {lr.score(x_test_scaled, y_test)}")
#print(f"predict : {lr.predict(x_test_scaled)}")

# 모델 초기화
svr_model = SVR(kernel='linear', C=1.0, epsilon=0.1)
# 모델 학습
svr_model.fit(x_train_scaled, y_train)
# 예측
#print(f" svr_model.predict (by test data) : {svr_model.predict(x_test_scaled)}")
print(f"score (by svr model, scaled train data) : {svr_model.score(x_train_scaled, y_train)}")

lasso_model = Lasso(alpha=0.5)
lasso_model.fit(x_train_scaled, y_train)

print(f"score (by lasso model, scaled train data) : {lasso_model.score(x_train_scaled, y_train)}")
print(f"score (by lasso model, scaled test data) : {lasso_model.score(x_test_scaled, y_test)}")
