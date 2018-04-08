import csv
import cv2
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation
from keras.layers.convolutional import Convolution2D

### Image preprocessing pipeline
def preprocess(image, color='RGB'):
    # by default use input image color space as is.
    img = image
    # convert to YUV color space
    if color == 'BGR':
        img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    elif color == 'RGB':
        img = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # trim image to only see section with road
    cropped_img = img[55:135,:,:]
    # rescale to nvidea model input size
    rescaled = cv2.resize(cropped_img,(200, 66), interpolation = cv2.INTER_LINEAR)
    return rescaled

### Build neural network model    
def PilotNet():
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x/127.5-1.0,input_shape=(66,200,3)))
    model.add(Convolution2D(24,5,5,subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    
    return model

### Load data, train and save convolutional neural network
if __name__ == '__main__':
    ### Load training data
    lines = []
    with open('./data/Track1-Centerlane/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []
    for line in lines:
        # read in images from center camera
        source_path = line[0]
        tokens = source_path.split('\\')
        filename = tokens[-1]
        local_path = "./data/Track1-Centerlane/IMG/" + filename
        image = cv2.imread(local_path)
        # add image to data set
        images.append(preprocess(image, color='BGR'))
        # add steering angles to data set
        measurements.append(float(line[3]))

    augmented_images = []
    augmented_measurements = []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(float(measurement))
        flipped_image = cv2.flip(image, 1)
        flipped_measurement = measurement * -1.0
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_measurement)

    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)
    
    model = PilotNet()
    
    ### Train neural network model
    history_object = model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=5)

    ### Save trained model
    model.save('model.h5')
    
    import matplotlib.pyplot as plt

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig('./examples/overfitting.jpg')
    plt.show()
