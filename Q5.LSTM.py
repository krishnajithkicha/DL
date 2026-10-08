import numpy as np
import tensorflow as tf

# Training text
text = "machine learning is interesting and machine learning is useful"

# Get unique characters
chars = sorted(set(text))

# Character to number
char_to_int = {c:i for i,c in enumerate(chars)}

# Number to character
int_to_char = {i:c for i,c in enumerate(chars)}

# Create input-output sequences
seq_length = 4
X = []
y = []

for i in range(len(text) - seq_length):
    X.append([char_to_int[c] for c in text[i:i+seq_length]])
    y.append(char_to_int[text[i+seq_length]])

X = np.array(X)
y = np.array(y)

# One-hot encode input
X = tf.keras.utils.to_categorical(X, num_classes=len(chars))

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(seq_length, len(chars))),
    tf.keras.layers.Dense(len(chars), activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X, y, epochs=100, verbose=0)

# Predict next character
input_text = "mach"

input_seq = np.array([
    [char_to_int[c] for c in input_text]
])

input_seq = tf.keras.utils.to_categorical(
    input_seq,
    num_classes=len(chars)
)

prediction = model.predict(input_seq, verbose=0)

predicted_char = int_to_char[np.argmax(prediction)]

print("Input:", input_text)
print("Predicted next character:", predicted_char)
