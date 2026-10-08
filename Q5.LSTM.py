import numpy as np
import tensorflow as tf

# -----------------------------
# 1. Training text
# -----------------------------

text = "machine learning is fun and machine learning is useful"

# Get characters
chars = sorted(set(text))

# Convert character to number
char_to_int = {c:i for i, c in enumerate(chars)}

# Convert number to character
int_to_char = {i:c for i, c in enumerate(chars)}

# -----------------------------
# 2. Create sequences
# -----------------------------

seq_length = 4

X = []
y = []

for i in range(len(text) - seq_length):
    sequence = text[i:i+seq_length]

    X.append([char_to_int[c] for c in sequence])
    y.append(char_to_int[text[i+seq_length]])

X = np.array(X)
y = np.array(y)

# One-hot encoding
X = tf.keras.utils.to_categorical(
    X,
    num_classes=len(chars)
)

# -----------------------------
# 3. Create LSTM model
# -----------------------------

model = tf.keras.Sequential([
    tf.keras.Input(shape=(seq_length, len(chars))),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(len(chars), activation="softmax")
])

# -----------------------------
# 4. Compile
# -----------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 5. Train
# -----------------------------

model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)

# -----------------------------
# 6. Function for prediction
# -----------------------------

def predict_next(word):

    sequence = []

    for c in word:
        sequence.append(char_to_int[c])

    sequence = np.array([sequence])

    sequence = tf.keras.utils.to_categorical(
        sequence,
        num_classes=len(chars)
    )

    prediction = model.predict(sequence, verbose=0)

    result = int_to_char[np.argmax(prediction)]

    print(word, "→", result)


# -----------------------------
# 7. Test multiple inputs
# -----------------------------

predict_next("mach")
predict_next("achi")
predict_next("fu")
predict_next("lear")

