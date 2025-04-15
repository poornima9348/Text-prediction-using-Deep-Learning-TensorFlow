import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense

# Read dataset
with open('/content/drive/MyDrive/IndiaUS.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

X = input_sequences[:, :-1]
y = np.array(tf.keras.utils.to_categorical(input_sequences[:, -1], num_classes=total_words))

# Create and compile the model
model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(GRU(100, return_sequences=True))
model.add(LSTM(150))
model.add(Dense(total_words, activation='softmax'))  # to convert predicted scores to probabilities
print(model.summary())

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Generate text
seed_text = "The two sides were not expected to"
next_words = 15

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = next(word for word, index in tokenizer.word_index.items() if index == predicted)
    seed_text += " " + output_word

print(seed_text)

loss, accuracy = model.evaluate(X, y)
print('Test Accuracy: {:.2f}%'.format(accuracy * 100))
