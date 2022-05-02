import numpy as np
import tensorflow_datasets as tfds

from matplotlib import pyplot
from os.path import exists

from keras.datasets import cifar10
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from tensorflow.keras.layers import Rescaling
from tensorflow.data import AUTOTUNE
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.utils import to_categorical

class cifar10_cnn:
    
    # @input datasetDirectory - Should be folder directory in same directory as this file, contains class directories of image data
    # @input imgDimensions - Tuple (dims, channels) where dims is the dimension of the images, channels num of channels of images
    def __init__(self, filename, datasetDirectory=None):
        self.modelFilename = filename
        self.datasetDirectory = datasetDirectory
        self.imgDimensions = (32, 3)
        self.model = None
    
    # Default to tensorflow Cifar10 if no dataset given
    def load_dataset(self):
        (trainX, trainY), (testX, testY) = cifar10.load_data()

        # One-Hot encoding
        trainY = to_categorical(trainY)
        testY = to_categorical(testY)
        
        return trainX, trainY, testX, testY
    
    # Load images into tensorflow datasets from directory
    def load_dataset_from_directory(self, batch_size=64, imgDimensions=None):
        # Define image dimensions here for some reason
        self.imgDimensions = imgDimensions if imgDimensions != None else self.imgDimensions
            
        # Load train/test from directory
        train_ds = image_dataset_from_directory(
            self.datasetDirectory,
            label_mode='categorical',
            validation_split=0.1,
            subset="training",
            seed=0,
            color_mode="grayscale",
            batch_size=batch_size,
            image_size=(32, 32)
        )
        test_ds = image_dataset_from_directory(
            self.datasetDirectory,
            label_mode='categorical',
            validation_split=0.1,
            subset="validation",
            seed=0,
            color_mode="grayscale",
            batch_size=batch_size,
            image_size=(32, 32)
        )
        
        # Convert to numpy
        train_numpy = tfds.as_numpy(train_ds)
        test_numpy = tfds.as_numpy(test_ds)
        
        # Split into X/Y
        trainX = np.array(list(map(lambda x: x[0], train_numpy)))
        trainX = np.concatenate(trainX[:])
        trainY = np.array(list(map(lambda x: x[1], train_numpy)))
        trainY = np.concatenate(trainY[:])
        
        testX = np.array(list(map(lambda x: x[0], test_numpy)))
        testX = np.concatenate(testX[:])
        testY = np.array(list(map(lambda x: x[1], test_numpy)))
        testY = np.concatenate(testY[:])
        
        return trainX, trainY, testX, testY
    
    
    def prep_pixels(self, train, test):
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')

        # Normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0

        return train_norm, test_norm
    
    def define_model(self, learning_rate=0.001, momentum=0.9):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, self.imgDimensions[1])))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))

        opt = SGD(learning_rate=learning_rate, momentum=momentum)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def fit_model(self, trainX, trainY, epochs=100, batch_size=64):
        self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)
    
    def save_model_to_file(self):
        self.model.save(self.modelFilename)
    
    def load_model_from_file(self):
        if exists(self.modelFilename):
            self.model = load_model(self.modelFilename)
    
    def run_test_harness(self, load_model=True, epochs=100, batch_size=64, learning_rate=0.001, momentum=0.9):
        # Load data and prep data
        if self.datasetDirectory == None:
            trainX, trainY, testX, testY = self.load_dataset()
        else:
            imgDimensions = (128, 1)
            trainX, trainY, testX, testY = self.load_dataset_from_directory(batch_size, imgDimensions)
            
        trainX, testX = self.prep_pixels(trainX, testX)
        
        # Load model from file if already trained (and desired to be used)
        if load_model:
            self.load_model_from_file()
        # If no model, train on train data then save
        if self.model == None:
            self.model = self.define_model(learning_rate, momentum)
            self.fit_model(trainX, trainY, epochs, batch_size)
            self.save_model_to_file()
       
        # Test accuracy
        _, acc = self.model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        
