#import tensorflow as tf
import keras
from keras import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.api.datasets import mnist
from keras.api.utils import to_categorical

# 1. 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. 모델 구성
model = Sequential([
    # input_shape ==> x, y, z (z는 채널 / 흑백이므로 1)
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 3. 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 모델 학습
model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

# 5. 성능 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
