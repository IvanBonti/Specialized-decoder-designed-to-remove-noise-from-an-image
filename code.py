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
noisly_sample_image = sample_image + noise_factor * np.random.randn(*(28,28))

plt.imshow(noisly_sample_image, cmap = "gray")

