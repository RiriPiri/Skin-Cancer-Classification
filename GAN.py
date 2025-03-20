import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Conv2D, Conv2DTranspose, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set parameters
IMG_SIZE = 64
CHANNELS = 3
LATENT_DIM = 100
BATCH_SIZE = 32
EPOCHS = 10000
DATASET_PATH = 'archive2/ISIC'

# Load and preprocess dataset
def load_dataset():
    images = []
    for file in os.listdir(DATASET_PATH):
        img = load_img(os.path.join(DATASET_PATH, file), target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img)
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        images.append(img)
    return np.array(images)

# Generator model
def build_generator():
    input = Input(shape=(LATENT_DIM,))
    x = Dense(8 * 8 * 256, activation='relu')(input)
    x = Reshape((8, 8, 256))(x)
    
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2DTranspose(CHANNELS, kernel_size=4, strides=2, padding='same', activation='tanh')(x)
    
    model = Model(input, x)
    return model

# Discriminator model
def build_discriminator():
    input = Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
    
    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(input)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(input, x)
    return model

# Compile models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# GAN model
discriminator.trainable = False
gan_input = Input(shape=(LATENT_DIM,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)

gan = Model(gan_input, gan_output)
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Training function
def train():
    X_train = load_dataset()
    real = np.ones((BATCH_SIZE, 1))
    fake = np.zeros((BATCH_SIZE, 1))

    for epoch in range(EPOCHS):
        # Train discriminator
        idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
        real_imgs = X_train[idx]

        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        g_loss = gan.train_on_batch(noise, real)

        if epoch % 100 == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")
            save_generated_images(epoch)

# Save generated images
def save_generated_images(epoch, examples=5):
    noise = np.random.normal(0, 1, (examples, LATENT_DIM))
    gen_imgs = generator.predict(noise)

    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale to [0, 1]

    fig, axs = plt.subplots(1, examples, figsize=(15, 5))
    for i in range(examples):
        axs[i].imshow(gen_imgs[i])
        axs[i].axis('off')
    plt.show()
    plt.savefig(f"generated_epoch_{epoch}.png")
    plt.close()

# Train the GAN
train()
