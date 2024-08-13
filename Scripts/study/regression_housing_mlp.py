import tensorflow as tf
import keras # keras를 바로 읽어야 인텔리센스가 잘 동작함.
import matplotlib.pyplot as plt
import pandas as pd

# 캘리포니아 주택 데이터셋 로드
california_housing_data = keras.datasets.california_housing.load_data()

# 데이터 분할
(x_train_full, y_train_full), (x_test, y_test) = california_housing_data
x_train, y_train = x_train_full[:-5000], y_train_full[:-5000]
x_valid, y_valid = x_train_full[-5000:], y_train_full[-5000:]

print(f"x_train.shape :{x_train.shape}")
print(f"y_train.shape :{y_train.shape}")

tf.random.set_seed(42)

norm_layer = keras.layers.Normalization(input_shape=x_train.shape[1:])

# 시퀀스 모델 생성
model = keras.Sequential([
    norm_layer,
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dense(1)
])

# 모델 요약
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mse', optimizer=optimizer, metrics=["RootMeanSquaredError"])
norm_layer.adapt(x_train)

# 모델 학습
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_valid, y_valid))

# 테스트 데이터 평가
mse_test, rmse_test = model.evaluate(x_test, y_test)
print(f"Test MSE: {mse_test}")
print(f"Test RMSE: {rmse_test}")

# 예측
x_new = x_test[:3]
y_pred = model.predict(x_new)
print(f"y_pred : {y_pred}")

# 학습 과정 시각화
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # 범위 설정
    plt.show()

plot_learning_curves(history)

# 예측 값과 실제 값 비교
y_pred_full = model.predict(x_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_full)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.axis('equal')
plt.axis('square')
plt.plot([-100, 100], [-100, 100], color='red')
plt.show()

# RMSE, MAE, R^2 등의 추가적인 평가 지표
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, y_pred_full)
mae = mean_absolute_error(y_test, y_pred_full)
r2 = r2_score(y_test, y_pred_full)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")
