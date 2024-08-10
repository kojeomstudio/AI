
import tensorflow as tf  # tensorflow, 텐서플로우로 바로 불러오면, lazy_load 되는게 있어 툴팁이나 인텔리센스가 정상 동작 안함.
import keras # keras로 바로 import 해서 사용하는게 편함.

# 데이터 시각화~
import matplotlib.pyplot as plt
import pandas as pd
# ~~

fashion_mnist = keras.datasets.fashion_mnist.load_data()

# split datas
(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist

x_train, y_train = x_train_full[:-5000], y_train_full[:-5000]
x_valid, y_valid = x_train_full[-5000:], y_train_full[-5000:]

print(f"x_train.shape :{x_train.shape}")
print(f"y_train.shape :{y_train.shape}")

# 정규화
x_train, x_valid, x_test = x_train / 255.0, x_valid / 255.0, x_test / 255.0

# 분류 라벨링
class_names = ["T-Shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


tf.random.set_seed(42)

# 케라스에서 제공하는 가장 기본적인 신경망(nn) 모델
model = keras.Sequential()

# layers
model.add(keras.layers.Input(shape=[28, 28]))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()

print(f"layers : {model.layers}")

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

result_history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))

pd.DataFrame(result_history.history).plot(figsize=(8,5), xlim=[0, 29], ylim=[0, 1], grid=True, xlabel="epoch",
                                          style=['r--', 'r--', 'b-', 'b-*'])
plt.show()