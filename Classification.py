"""Convolutional Neural Network Estimator for DeepScores Classification, built with Tensorflow
"""

import argparse
import sys, os

import tensorflow as tf
import Classification_BatchDataset
import TensorflowUtils as utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

FLAGS = None



def create_deepscores_cnn_model(input_shape, nr_classes):
    model = Sequential([
    #input
    Conv2D(32, (3,3), activation='relu', padding="same", strides=(1, 1), use_bias=True,  bias_initializer='zeros', input_shape=input_shape),
    MaxPool2D((2,2), padding="same"),
    Dropout(0.1),
    Conv2D(64, (3,3), activation='relu', padding="same", strides=(1, 1), use_bias=True,  bias_initializer='zeros'),
    MaxPool2D((2,2), padding="same"),
    Dropout(0.1),
    Conv2D(128, (3,3), activation='relu', padding="same", strides=(1, 1), use_bias=True,  bias_initializer='zeros'),
    MaxPool2D((2,2), padding="same"),
    Dropout(0.1),
    Conv2D(256, (3,3), activation='relu', padding="same", strides=(1, 1), use_bias=True,  bias_initializer='zeros'),
    MaxPool2D((2,2), padding="same"),
    Dropout(0.1),
    Conv2D(512, (3,3), activation='relu', padding="same", strides=(1, 1), use_bias=True,  bias_initializer='zeros'),
    MaxPool2D((2,2), padding="same"),
    Dropout(0.1),
    Flatten(),
    Dense(1024),
    Dense(nr_classes)
    ])
    return model


def main(FLAGS):
    dataset_dir = FLAGS.data_dir
    batch_size=FLAGS.batch_size
    epochs=FLAGS.epochs
    mode=FLAGS.mode
    train_dir = os.path.join(dataset_dir, 'train')
    validation_dir = os.path.join(dataset_dir, 'validate')
    test_dir = os.path.join(dataset_dir, 'test')
    classes= os.listdir(train_dir)


    image_generator = ImageDataGenerator(rescale=1/255.)
    train_gn = image_generator.flow_from_directory(directory=train_dir, batch_size=batch_size, shuffle=True,
                    target_size=(120, 220), classes=classes)
    validation_gn = image_generator.flow_from_directory(directory=validation_dir, batch_size=batch_size, shuffle=True,
                    target_size=(120, 220), classes=classes)
    test_gn = image_generator.flow_from_directory(directory=test_dir, batch_size=batch_size, shuffle=True,
                    target_size=(120, 220), classes=classes)


    model = create_deepscores_cnn_model((220, 120, 3), len(classes))
    model.compile(optimizer= 'adam', loss=CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.summary()

    history = model.fit(train_gn, epochs=epochs, validation_data=validation_gn)
    results = model.evaluate(test_gn, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                      default='dataset',
                      help='Directory for storing input data')
    parser.add_argument('--batch_size', type=int,
                      default=128)
    parser.add_argument('--epochs', type=int,
                    default=50)
    parser.add_argument('-mode', default='train', type=str)

    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
