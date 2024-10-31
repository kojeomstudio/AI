import numpy as np
import pandas as pd

from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # 표준화 (데이터 전처리))
import matplotlib.pyplot as plt # 그래프

# CSV 파일 불러오기
file_path = './Datas/mnist.csv'
raw_data = pd.read_csv(file_path)

# 데이터 확인
#print("==== raw_data.head() ====")
#print(f"{raw_data.head()}")

# 데이터 정보 확인
#print(f"raw_data.info() : {raw_data.info()}")

# 기초 통계량 확인
#print("raw_data.describe")
#print(f"{raw_data.describe}")

print(f"columns : {raw_data.columns}")
print(f"raw_data shape : {raw_data.shape}")

x = raw_data.drop('label', axis=1) # (=학습에 이용)
y = raw_data['label'] # (=정답)

print(f"x's shape : {x.shape}")
print(f"y's shape : {y.shape}")

# 데이터 나누기
# (훈련, 테스트, 검증)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 표준화
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)  # 검증 데이터 표준화
x_test_scaled = scaler.transform(x_test)  # 테스트 데이터 표준화

# 모델 정의
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(x_train_scaled.shape[1],)),
    keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(x_train_scaled, y_train, epochs=5, validation_data=(x_val_scaled, y_val))

# 모델 평가
loss, accuracy = model.evaluate(x_test_scaled, y_test)
print(f"Test accuracy: {accuracy}")
