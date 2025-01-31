# 세익스피어 문장을 이용한 rnn 테스트 코드.

import tensorflow as tf
import keras


#############################################################################
def to_dataset(sequence, length,  shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))

    if shuffle:
        ds = ds.shuffle(buffer_size=100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)

def to_dataset_for_stateful_rnn(seqeunce, length):
    ds = tf.data.Dataset.from_tensor_slices(seqeunce)
    ds = ds.window(length + 1, shift=length, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(length + 1)).batch(1)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)   
##############################################################################

shakespeare_url = "https://homl.info/shakespeare" #...
filepath = keras.utils.get_file("shakespeare.txt", shakespeare_url)
print(f"downloaded file path : {filepath}")

with open(filepath) as f:
    shakespeare_text = f.read()
    #print(f" 세익스피어 데이터 원문 : {shakespeare_text}")
    text_to_vec_layer = keras.layers.TextVectorization(split="character", standardize="lower")
    text_to_vec_layer.adapt([shakespeare_text])
    
    encoded = text_to_vec_layer([shakespeare_text])[0]
    encoded -= 2 # 토큰 0(패딩), 1(미식별 문자)을 사용하지 않으므로 무시 처리.
    n_tokens = text_to_vec_layer.vocabulary_size() - 2 # 고유한 문자 개수 = 39
    dataset_size = len(encoded)
    print(f"dataset_size : {dataset_size}")

    # 데이터 준비
    length = 100
    tf.random.set_seed(42)
    train_set = to_dataset(encoded[:1_000_000], length=length, shuffle=True, seed=42)
    valid_set = to_dataset(encoded[1_000_000:1_060_000], length=length)
    test_set = to_dataset(encoded[1_060_000:], length=length)

    # 모델 생성
    model = keras.Sequential([
        keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
        keras.layers.GRU(128, return_sequences=True),
        keras.layers.Dense(n_tokens, activation="softmax")
    ])

    model_file_path = "my_shakespeare_model.keras"

    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    model_ckpt = keras.callbacks.ModelCheckpoint(
        model_file_path, monitor="val_accuracy", save_best_only=True)
    history = model.fit(train_set, validation_data=valid_set, epochs=10, callbacks=[model_ckpt])

    shakespeare_model = keras.Sequential([
        text_to_vec_layer,
        keras.layers.Lambda(lambda X: X -2), model
    ])

    # test.
    y_proba = shakespeare_model.predict(["To be or not to b"])[0, -1]
    y_pred = tf.argmax(y_proba)
    print(f"text_to_vec_layer -> prediction vocabulary : {text_to_vec_layer.get_vocabulary()[y_pred + 2]}")
    



