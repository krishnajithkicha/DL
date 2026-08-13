# Denoising Autoencoder for MNIST
# --------------------------------

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------
# 1. Load MNIST dataset
# ---------------------------------------------------------
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Add channel dimension: (samples, 28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)
x_test  = np.expand_dims(x_test, axis=-1)

print("Training images:", x_train.shape)
print("Testing images :", x_test.shape)


# ---------------------------------------------------------
# 2. Add random Gaussian noise
# ---------------------------------------------------------
noise_factor = 0.5

# Generate Gaussian noise
train_noise = np.random.normal(
    loc=0.0,
    scale=1.0,
    size=x_train.shape
)

test_noise = np.random.normal(
    loc=0.0,
    scale=1.0,
    size=x_test.shape
)

# Add noise
x_train_noisy = x_train + noise_factor * train_noise
x_test_noisy = x_test + noise_factor * test_noise

# Clip values so they remain in [0, 1]
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

print("Gaussian noise added.")


# ---------------------------------------------------------
# 3. Construct the Convolutional Autoencoder
# ---------------------------------------------------------

# Encoder
encoder = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D(
        (2, 2),
        padding="same"
    ),

    layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D(
        (2, 2),
        padding="same"
    ),

    layers.Conv2D(
        16, (3, 3),
        activation="relu",
        padding="same"
    )
])


# Decoder
decoder = models.Sequential([
    layers.Conv2D(
        16, (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.UpSampling2D((2, 2)),

    layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.UpSampling2D((2, 2)),

    layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same"
    ),

    # Sigmoid ensures output is in [0, 1]
    layers.Conv2D(
        1, (3, 3),
        activation="sigmoid",
        padding="same"
    )
])


# Complete autoencoder
autoencoder = models.Sequential([
    encoder,
    decoder
])

# ---------------------------------------------------------
# 4. Compile the model
# ---------------------------------------------------------
autoencoder.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["mse"]
)

autoencoder.summary()


# ---------------------------------------------------------
# 5. Train the Denoising Autoencoder
# ---------------------------------------------------------
history = autoencoder.fit(
    x_train_noisy,
    x_train,                    # Clean images are targets
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)


# ---------------------------------------------------------
# 6. Reconstruct / Denoise test images
# ---------------------------------------------------------
decoded_imgs = autoencoder.predict(
    x_test_noisy,
    batch_size=128
)

print("Denoising completed.")


# ---------------------------------------------------------
# 7. Display Original, Noisy and Reconstructed Images
# ---------------------------------------------------------
n = 10

plt.figure(figsize=(20, 6))

for i in range(n):

    # Original image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Noisy image
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].squeeze(), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    # Denoised image
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs[i].squeeze(), cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 8. Plot training and validation loss
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Denoising Autoencoder Training")
plt.legend()
plt.grid(True)

plt.show()


# ---------------------------------------------------------
# 9. Quantitative reconstruction quality
# ---------------------------------------------------------
mse = np.mean(
    np.square(x_test - decoded_imgs)
)

noisy_mse = np.mean(
    np.square(x_test - x_test_noisy)
)

print("\nReconstruction Quality")
print("----------------------")
print(f"Noisy Image MSE     : {noisy_mse:.6f}")
print(f"Denoised Image MSE  : {mse:.6f}")

if mse < noisy_mse:
    print("Result: The autoencoder successfully reduced the noise.")
else:
    print("Result: Increase training epochs or adjust the model.")
