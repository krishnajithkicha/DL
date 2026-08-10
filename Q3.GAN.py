# ==========================================
# Generative Adversarial Network (GAN)
# MNIST Handwritten Digit Generation
# ==========================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers

# ------------------------------------------
# 1. Load and Preprocess MNIST Dataset
# ------------------------------------------

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from [0, 255] to [-1, 1]
x_train = x_train.astype("float32")
x_train = (x_train - 127.5) / 127.5

# Add channel dimension: (60000, 28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)

print("Dataset shape:", x_train.shape)

# Create batches
BUFFER_SIZE = 60000
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# ------------------------------------------
# 2. Design the Generator
# ------------------------------------------

def build_generator():

    model = tf.keras.Sequential([
        
        # Input: random noise vector
        layers.Dense(7 * 7 * 256, use_bias=False,
                     input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Convert vector into 7x7x256 feature map
        layers.Reshape((7, 7, 256)),

        # 7x7 -> 14x14
        layers.Conv2DTranspose(
            128, (5, 5),
            strides=(1, 1),
            padding="same",
            use_bias=False
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # 14x14 -> 28x28
        layers.Conv2DTranspose(
            64, (5, 5),
            strides=(2, 2),
            padding="same",
            use_bias=False
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Output: 28x28x1
        layers.Conv2DTranspose(
            1, (5, 5),
            strides=(2, 2),
            padding="same",
            activation="tanh"
        )
    ])

    return model


generator = build_generator()

generator.summary()


# ------------------------------------------
# 3. Design the Discriminator
# ------------------------------------------

def build_discriminator():

    model = tf.keras.Sequential([

        layers.Conv2D(
            64, (5, 5),
            strides=(2, 2),
            padding="same",
            input_shape=[28, 28, 1]
        ),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(
            128, (5, 5),
            strides=(2, 2),
            padding="same"
        ),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),

        # Output probability: real/fake
        layers.Dense(1)
    ])

    return model


discriminator = build_discriminator()

discriminator.summary()


# ------------------------------------------
# 4. Define Loss Functions and Optimizers
# ------------------------------------------

cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)


def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(
        tf.ones_like(real_output),
        real_output
    )

    fake_loss = cross_entropy(
        tf.zeros_like(fake_output),
        fake_output
    )

    total_loss = real_loss + fake_loss

    return total_loss


def generator_loss(fake_output):

    return cross_entropy(
        tf.ones_like(fake_output),
        fake_output
    )


generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5
)

discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0002,
    beta_1=0.5
)


# ------------------------------------------
# 5. Training Setup
# ------------------------------------------

EPOCHS = 50

noise_dim = 100

num_examples_to_generate = 16

# Fixed noise for displaying progress
seed = tf.random.normal(
    [num_examples_to_generate, noise_dim]
)

# Store losses
generator_losses = []
discriminator_losses = []


# ------------------------------------------
# 6. Training Step
# ------------------------------------------

@tf.function
def train_step(images):

    # Generate random noise
    noise = tf.random.normal(
        [BATCH_SIZE, noise_dim]
    )

    with tf.GradientTape() as gen_tape, \
         tf.GradientTape() as disc_tape:

        # Generate fake images
        generated_images = generator(
            noise,
            training=True
        )

        # Discriminator predictions
        real_output = discriminator(
            images,
            training=True
        )

        fake_output = discriminator(
            generated_images,
            training=True
        )

        # Calculate losses
        gen_loss = generator_loss(fake_output)

        disc_loss = discriminator_loss(
            real_output,
            fake_output
        )

    # Calculate gradients
    gradients_of_generator = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables
    )

    gradients_of_discriminator = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    # Update weights
    generator_optimizer.apply_gradients(
        zip(
            gradients_of_generator,
            generator.trainable_variables
        )
    )

    discriminator_optimizer.apply_gradients(
        zip(
            gradients_of_discriminator,
            discriminator.trainable_variables
        )
    )

    return gen_loss, disc_loss


# ------------------------------------------
# 7. Generate and Display Images
# ------------------------------------------

def generate_and_display_images(epoch):

    predictions = generator(
        seed,
        training=False
    )

    fig = plt.figure(figsize=(6, 6))

    for i in range(predictions.shape[0]):

        plt.subplot(4, 4, i + 1)

        plt.imshow(
            predictions[i, :, :, 0] * 127.5 + 127.5,
            cmap="gray"
        )

        plt.axis("off")

    plt.suptitle(
        "Generated Images - Epoch " + str(epoch)
    )

    plt.tight_layout()
    plt.show()


# ------------------------------------------
# 8. Train GAN
# ------------------------------------------

def train(dataset, epochs):

    for epoch in range(1, epochs + 1):

        gen_loss_total = 0
        disc_loss_total = 0
        batch_count = 0

        for image_batch in dataset:

            gen_loss, disc_loss = train_step(
                image_batch
            )

            gen_loss_total += float(gen_loss)
            disc_loss_total += float(disc_loss)

            batch_count += 1

        # Average loss
        avg_gen_loss = gen_loss_total / batch_count
        avg_disc_loss = disc_loss_total / batch_count

        generator_losses.append(avg_gen_loss)
        discriminator_losses.append(avg_disc_loss)

        print(
            f"Epoch {epoch}/{epochs} "
            f"| Generator Loss: {avg_gen_loss:.4f} "
            f"| Discriminator Loss: {avg_disc_loss:.4f}"
        )

        # Display generated digits every 10 epochs
        if epoch % 10 == 0:
            generate_and_display_images(epoch)


# ------------------------------------------
# 9. Start Training
# ------------------------------------------

train(train_dataset, EPOCHS)


# ------------------------------------------
# 10. Generate Final 16 Images
# ------------------------------------------

noise = tf.random.normal(
    [16, noise_dim]
)

generated_images = generator(
    noise,
    training=False
)

plt.figure(figsize=(6, 6))

for i in range(16):

    plt.subplot(4, 4, i + 1)

    plt.imshow(
        generated_images[i, :, :, 0] * 127.5 + 127.5,
        cmap="gray"
    )

    plt.axis("off")

plt.suptitle("Final Generated MNIST Digits")
plt.tight_layout()
plt.show()


# ------------------------------------------
# 11. Plot Generator Loss
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    generator_losses,
    color="blue"
)

plt.title("Generator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()

plt.show()


# ------------------------------------------
# 12. Plot Discriminator Loss
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    discriminator_losses,
    color="red"
)

plt.title("Discriminator Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()

plt.show()


# ------------------------------------------
# 13. Save Trained Generator
# ------------------------------------------

generator.save("mnist_gan_generator.keras")

print("GAN training completed!")
print("Generator saved as mnist_gan_generator.keras")
