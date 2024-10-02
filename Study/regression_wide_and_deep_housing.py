import tensorflow as tf
import keras # keras를 바로 읽어야 인텔리센스가 잘 동작함.
import matplotlib.pyplot as plt
import pandas as pd

# wide & deep 방식으로 딥러닝 학습
# -> 2016 헝쯔 청의 논문.

# 캘리포니아 주택 데이터셋 로드
california_housing_data = keras.datasets.california_housing.load_data()

# 데이터 분할
(x_train_full, y_train_full), (x_test, y_test) = california_housing_data
x_train, y_train = x_train_full[:-5000], y_train_full[:-5000]
x_valid, y_valid = x_train_full[-5000:], y_train_full[-5000:]

print(f"x_train.shape :{x_train.shape}")
print(f"y_train.shape :{y_train.shape}")

tf.random.set_seed(42)

normalization_layer = keras.layers.Normalization()

# 활성화 함수를 적용 함으로서, 비선형으로 변환이 된다.
hidden_layer1 = keras.layers.Dense(30, activation="relu")
hidden_layer2 = keras.layers.Dense(30, activation="relu")
concat_layer = keras.layers.Concatenate()
output_layer = keras.layers.Dense(1)

input_ = keras.layers.Input(shape=x_train.shape[1:])
normalized = normalization_layer(input_)

out_hidden1 = hidden_layer1(normalized)
out_hidden2 = hidden_layer2(out_hidden1)
concat = concat_layer([normalized, out_hidden2])
output = output_layer(concat)

model = keras.Model(inputs=[input_], outputs=[output])

# 모델 요약
model.summary()

optimizer = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mse', optimizer=optimizer, metrics=["RootMeanSquaredError"])
normalization_layer.adapt(x_train)

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

model.save("./Models/wide_and_deep_regression_housing_model.keras")
model.save_weights("./Mddels/wide_and_deep_regression_housing_weight.weights.h5")