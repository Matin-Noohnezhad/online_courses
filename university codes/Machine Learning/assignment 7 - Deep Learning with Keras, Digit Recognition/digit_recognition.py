from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import numpy
from keras import backend as K
K.set_image_dim_ordering('th')

# Take a look at two of the images.
plt.subplot(221)
plt.imshow(X_train[50], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[68], cmap=plt.get_cmap('gray'))
plt.show()

# Load images of handwritten-digits
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# The size of X_train is N by Width by Height (you can use shape to get the sizes).
# Use reshape to change the size of X_train to N by 1 by Width by Height. Similarly do the same thing for X_test 

X_train = ...
X_test = ...

# each pixle is between 0 and 255, normalize the image by making the pixles between 0 and 1 by dividing by 255. 

X_train = ...
X_test = ...

# Make y a hot vector. by using np_utils.to_categorical

y_train = ...
y_test = ...

number_of_classes = y_test.shape[1]

#Create a multi-layer convolutional neural network in the following way: 
# For the first layer create 50 convolutional networks with window sizes of 5 by 5, and ReLu for the activation. 
# Add a max pooling layer with a window of 2 by 2. 
# For the second layer create 20 colvolutional networks with window sizes of 3 by 3, ReLu for the activation. 
# Add a max pooling layer with a window of 2 by 2. 
# For the third layer create 10 colvolutional networks with window sizes of 2 by 2, ReLu for the activation. 
# Add a max pooling layer with a window of 2 by 2. 
# Flatten the last layer. 
# For the forth layer use a Dense layer of size 60 with ReLu for activation
# For the fifth layer use a Dense layer of size number_of_classes with softmax for activation 
# Use categorical_crossentropy for loss and adam for optimizer, 10 for epochs and 100 for batch_size

print X_train.shape

model = Sequential()
model.add(Conv2D(...))
model.add(MaxPooling2D(...))
...
...
model.compile(...)
model.fit(...)

# Compute the accuracy of the model on X_test and y_test
...

# Add dropouts of 0.20 right after your max pooling layer. Do you get a better result? 
# What accuracy do you get when using RMSprop for your optimizer? 
# What accuracy do you get when using SGD for your optimizer? 