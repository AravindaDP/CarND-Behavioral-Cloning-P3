import csv
import cv2
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2

from sklearn.utils import shuffle

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
    model.add(Convolution2D(24,5,5,subsample=(2,2), W_regularizer = l2(1e-6)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2), W_regularizer = l2(1e-6)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2), W_regularizer = l2(1e-6)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3, W_regularizer = l2(1e-6)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3, W_regularizer = l2(1e-6)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(100, W_regularizer = l2(1e-6)))
    model.add(Dense(50, W_regularizer = l2(1e-6)))
    model.add(Dense(10, W_regularizer = l2(1e-6)))
    model.add(Dense(1, W_regularizer = l2(1e-6)))

    model.compile(optimizer='adam', loss='mse')
    
    return model

### Load training data
def load_data(path, augmented_images, augmented_measurements, correction = 0.2):
    images = []
    measurements = []
    with open(path + "driving_log.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # read in images from center, left and right cameras
            for i in range(3):
                source_path = row[i]
                tokens = source_path.split('\\')
                filename = tokens[-1]
                local_path = path + "IMG/" + filename
                image = cv2.imread(local_path)
                # add images to data set
                images.append(preprocess(image, color='BGR'))
            steering_center = float(row[3])

            # create adjusted steering measurements for the side camera images
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            # add steering angles to data set
            measurements.append(steering_center)
            measurements.append(steering_left)
            measurements.append(steering_right)

    ### Data augmentation
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        image_flipped = cv2.flip(image, 1)
        measurement_flipped = measurement * -1.0
        augmented_images.append(image_flipped)
        augmented_measurements.append(measurement_flipped)

    return augmented_images, augmented_measurements

### Load data, train and save convolutional neural network
if __name__ == '__main__':
    images = []
    measurements = []

    correction = 0.2 # this is a parameter to tune
    images, measurements = load_data("./data/Track1-Centerlane/",images, measurements, correction = correction)
    images, measurements = load_data("./data/Track1-Recovery/",images, measurements, correction = correction)
    images, measurements = load_data("./data/Track1-Counter-clock/",images, measurements, correction = correction)

    X_train = np.array(images)
    y_train = np.array(measurements)
    
    X_train, y_train = shuffle(X_train, y_train)

    model = PilotNet()
    
    ### Train neural network model
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

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
    plt.savefig('./examples/model_training.jpg')
    plt.show()
