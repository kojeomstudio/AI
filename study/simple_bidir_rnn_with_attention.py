import tensorflow as tf
import keras
from pathlib import Path
import numpy as np

url = "https://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip"
path = keras.utils.get_file("spa-eng.zip", origin=url, cache_dir="datasets", extract=True)
print(f"downloaded text file path : {path}")

target_path = (Path(path) / "spa-eng" / "spa.txt")
print(f"target file path : {target_path}")

text = target_path.read_text()

text = text.replace('i', '').replace('¿', '') # 스페인어 역물음표
pairs = [line.split('\t') for line in text.splitlines()]
np.random.shuffle(pairs)

sentences_en, sentences_es = zip(*pairs)

#test
for i in range(3):
    print(f"convert eng : {sentences_en[i]} to spanish : {sentences_es[i]}")

vocab_size = 1000
max_length = 50

text_vec_layer_en = keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length
)

text_vec_layer_es = keras.layers.TextVectorization(
    vocab_size, output_sequence_length=max_length
)

text_vec_layer_en.adapt(sentences_en)
text_vec_layer_es.adapt([f"startofseq {s} endofseq" for s in sentences_es])

x_train = tf.constant(sentences_en[:100_000])
x_valid = tf.constant(sentences_en[100_000:])
x_train_dec = tf.constant([f"startofseq {s}" for s in sentences_es[:100_000]])
x_valid_dec = tf.constant([f"startofseq {s}" for s in sentences_es[100_000:]])
y_train = text_vec_layer_es([f"{s} endofseq" for s in sentences_es[:100_000]])
y_valid = text_vec_layer_es([f"{s} endofseq" for s in sentences_es[100_000:]])

encoder_inputs = keras.layers.Input(shape=[], dtype=tf.string)
decoder_inputs = keras.layers.Input(shape=[], dtype=tf.string)

embed_size = 128
encoder_input_ids = text_vec_layer_en(encoder_inputs)
decoder_input_ids = text_vec_layer_es(decoder_inputs)

encoder_embedding_layer = keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
decoder_embedding_layer = keras.layers.Embedding(vocab_size, embed_size, mask_zero=True)
encoder_embeddings = encoder_embedding_layer(encoder_input_ids)
decoder_embeddings = decoder_embedding_layer(decoder_input_ids)

# 양방향 rnn 
encoder = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True, return_state=True))
encoder_outputs, *encoder_state = encoder(encoder_embeddings)

encoder_state = [keras.layers.concatenate(encoder_state[::2], axis=-1), # 단기 상태
                 keras.layers.concatenate(encoder_state[1::2], axis=-1)] # 장기 상태

decoder = keras.layers.LSTM(512, return_sequences=True)
decoder_outputs = decoder(decoder_embeddings, initial_state=encoder_state)

# attention
attention_layer = keras.layers.Attention()
attetion_outputs = attention_layer([decoder_outputs, encoder_outputs])

output_layer = keras.layers.Dense(vocab_size, activation='softmax')
y_proba = output_layer(attetion_outputs)

model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=[y_proba])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam',
              metrics=['accuracy'])
model.fit((x_train, x_train_dec), y_train, epochs=10, validation_data=((x_valid, x_valid_dec), y_valid))

def translate_helper(sentence_en):
    transloation = ""

    for word_idx in range(max_length):
        x = np.array([sentence_en])
        x_dec = np.array(["startofseq" + transloation])
        
        y_proba_local = model.predict((x, x_dec))[0, word_idx]
        predicted_word_id = np.argmax(y_proba_local)
        predicted_word = text_vec_layer_es.get_vocabulary()[predicted_word_id]

        if predicted_word == 'endofseq':
            break
        transloation += ' ' + predicted_word
    return transloation.strip()

result = translate_helper("I like Soccer")
print(f"translated : {result}")