
"""
ECE196 Face Recognition Project
Author: W Chen

Adapted from: https://keras.io/getting-started/functional-api-guide/

Modify this code to write a LeNet with the following requirements:
* Input dimensions: 32x32x1 - Input
* C1: Convolutional Layer - Conv2D
    number of filters: 6
    kernel size: 5x5
    strides: 1 both horizontally and vertically (Set by default.)
    activation function: sigmoid
    output: 6 layers of 28x28 feature maps (Do not need to specify in function.)
* S2: Max Pooling Layer - MaxPooling2D
    pooling size: 2x2
    strides: 2 both horizontally and vertically
    output: 6 layers of 14x14 feature maps (Do not need to specify in function.)
* C3: Convolutional Layer - Conv2D
    number of filters: 16
    kernel size: 5x5
    strides: 1 both horizontally and vertically
    activation function: sigmoid
    output: 16 layers of 10x10 feature maps(Do not need to specify in function.)
* S4: Max Pooling Layer - MaxPooling2D
    pooling size: 2x2
    strides: 2 both horizontally and vertically
    output: 16 layers of 5x5 feature maps (Do not need to specify in function.)
* C5: Convolutional Layer - Conv2D
    number of filters: 120
    kernel size: 5x5
    strides: 1 both horizontally and vertically
    activation function: sigmoid
    output: 120 layers of 1x1 feature maps(Do not need to specify in function.)
* F6: Fully Connected Layer - Dense
    units: 84
    activation function: tanh
    output 84-dimensional vector (This is specified through units.)
* F7: Fully Connected Layer - Dense
    units: 10
    activation function: softmax
    output 10-dimensional vector (This is specified through units.)
"""
# TODO: Import other layers as necessary. (Conv2D, MaxPooling2D)

from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.convolutional  import Conv2D
from keras.layers.pooling import MaxPooling2D

# TODO: Currently, sets input dimension to be 784x1. Change to 32x32x1
inputs = Input(shape=(32,32,1))

# A layer instance is callable on a tensor, and returns a tensor
# This createa a model that includes the Input layer and three Dense layers
conv1 = Conv2D(6, 5, activation='sigmoid')(inputs)
pool2 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv1)
conv3 = Conv2D(16, kernel_size=(5, 5), activation='sigmoid')(pool2)
pool4 = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(conv3)
conv5 = Conv2D(120, kernel_size=(5, 5), activation='sigmoid')(pool4)

F6 = Dense(84,activation='tanh')(conv5)
F7 = Dense(10,activation='softmax')(F6)
predictions = F7 

model = Model(inputs=inputs, outputs=predictions)

# Prints model architecture
model.summary()

