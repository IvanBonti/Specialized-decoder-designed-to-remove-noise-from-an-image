#Specialized decoder designed to remove noise from an image

#Import packages
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import random 

#Importing our dataset
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

#Visualization of the dataset
x_train.shape
x_test.shape

#Selecting images randomly.
i = random.randint(1, 60000) #select 1 image from 60.000 images for train
plt.imshow(x_train[i], cmap = 'gray')
 
label = y_train[i]
print(label)

#Adding noise to our images
#Normalizing
x_train = x_train/255
x_test = x_test / 255

#Adding noise
added_noise = np.random.randn(*(28,28))
noise_factor = 0.3
added_noise = noise_factor * np.random.randn(*(28,28))

#visualization the created image
plt.imshow(added_noise)

#Select an randomly image and add noise 
noise_factor = 0.2
sample_image = x_train[101]
noisy_sample_image = sample_image + noise_factor * np.random.randn(*(28,28))

plt.imshow(noisy_sample_image, cmap = "gray")

#We can see by executing these two lines that the maximum and minimum values of the range are not between 0 and 1.
noisy_sample_image.max()
noisy_sample_image.min()

#To normalize, we use the clip function
noisy_sample_image = np.clip (noisy_sample_image,0.,1.)

plt.imshow(noisy_sample_image, cmap = 'gray')

#Apply the same operation to all the images
x_train_noisy=[]
noise_factor=0.2

for sample_image in  x_train:
    sample_image_noisy = sample_image + noise_factor *np.random.randn(*(28,28))
    sample_image_noisy = np.clip(sample_image_noisy,0.,1.)
    x_train_noisy.append(sample_image_noisy)

#convert our dataset in a matrix
x_train_noisy = np.array(x_train_noisy)

#Visualize the shape of the dataset
x_train_noisy.shape

#Checking that all our images have a range between 0 and 1, taking one at random
plt.imshow(x_train_noisy[140], cmap = "gray")


#Applying all the same operations to our test set
x_test_noisy=[]
noise_factor=0.4

for sample_image in  x_test:
    sample_image_noisy = sample_image + noise_factor *np.random.randn(*(28,28))
    sample_image_noisy = np.clip(sample_image_noisy,0.,1.)
    x_test_noisy.append(sample_image_noisy)

x_test_noisy = np.array(x_test_noisy)

x_test_noisy.shape

#Verifying that the images have more noise than those in the training set and have been processed, by printing one at random
plt.imshow(x_test_noisy[14], cmap = "gray")

#creating a model
autoencoder = tf.keras.models.Sequential()

#setting up a convolutional layer
#Adding a 2D convolutional layer with 16 filters of size 3x3, using a stride of 1 and input shape corresponding to the shape of the images:z  
autoencoder.add(tf.keras.layers.Conv2D(16,(3,3),strides = 1, padding="same", input_shape=(28,28,1)))
autoencoder.add(tf.keras.layers.MaxPooling2D((2,2),padding="same"))

autoencoder.add(tf.keras.layers.Conv2D(8,(3,3),strides = 1, padding="same"))
autoencoder.add(tf.keras.layers.MaxPooling2D((2,2),padding="same"))

#Encoded image.
autoencoder.add(tf.keras.layers.Conv2D(8,(3,3),strides = 1, padding="same"))

#Setting up a decoder
autoencoder.add(tf.keras.layers.UpSampling2D((2,2)))
autoencoder.add(tf.keras.layers.Conv2DTranspose(8,(3,3),strides = 1, padding="same"))

autoencoder.add(tf.keras.layers.UpSampling2D((2,2)))
autoencoder.add(tf.keras.layers.Conv2DTranspose(1,(3,3),strides = 1, activation='sigmoid', padding = "same"))


#compiled
autoencoder.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001))

#Visualization of our network
autoencoder.summary()

#training
autoencoder.fit(x_train_noisy.reshape(-1,28,28,1),
                x_train.reshape(-1,28,28,1),
                epochs=10,
                batch_size=200)


#Evaluating the model
denoissed_images = autoencoder.predict(x_test_noisy[:15].reshape(-1,28,28,1))
denoissed_images.shape

fig, axes = plt.subplots(nrows=2,ncols=15, figsize=(30,6))
for images , row in zip([x_test_noisy[:15], denoissed_images],axes):
    for img, ax in zip(images,row):
        ax.imshow(img.reshape((28,28)), cmap = 'gray')

