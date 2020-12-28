

import argparse
import sys, os

import imageio
import tensorflow as tf
import Classification_BatchDataset
import TensorflowUtils as utils
import pickle
import time

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Input, Lambda,BatchNormalization, ReLU
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

import numpy as  np
import numpy.random as rng

FLAGS = None


def loadimgs(path,n = 0):
    '''
    path => Path of train directory or test directory
    '''
    X=[]
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n
    # we load every alphabet seperately so we can isolate them later
    for alphabet in os.listdir(path):
        print("loading alphabet: " + alphabet)
        lang_dict[alphabet] = [curr_y,None]
        alphabet_path = os.path.join(path,alphabet)
        # every letter/category has it's own column in the array, so  load seperately
        for letter in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, letter)
            category_images=[]
            letter_path = os.path.join(alphabet_path, letter)
            # read all the images in the current category
            dirlist = os.listdir(letter_path)
            if len(dirlist)==100:
                for filename in dirlist:
                    image_path = os.path.join(letter_path, filename)
                    image = imageio.imread(image_path)
                    category_images.append(image)
                    # print(len(category_images))
                    y.append(curr_y)

                try:
                    uu = np.stack(category_images)
                    X.append(uu)
                # edge case  - last one
                except ValueError as e:
                    print(e)
                    print("error - category_images:", category_images)
                    print(letter)
                curr_y += 1
                lang_dict[alphabet][1] = curr_y - 1
    y = np.vstack(y)
    X = np.stack(X)
    return X,y,lang_dict





# def initialize_weights(shape, name=None):
#     """
#         The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
#         suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
#     """
#     return tf.random.normal(shape, mean = 0.0, stddev = 0.01)
#
# def initialize_bias(shape, name=None):
#     """
#         The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
#         suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
#     """
#     return tf.random.normal(shape, mean = 0.5, stddev = 0.01)


def get_siamese_model(input_shape):
    """
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    """

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    initialize_weights = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
    initialize_bias = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01)
    # Convolutional Neural Network
    model = Sequential([
    Conv2D(64, (10,10),  input_shape=input_shape,
                   kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)),
    BatchNormalization(),
    ReLU(),
    MaxPool2D(),
    Conv2D(128, (7,7),
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)),
    BatchNormalization(),
    ReLU(),
    MaxPool2D(),
    Conv2D(128, (4,4),  kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)),
    BatchNormalization(),
    ReLU(),
    MaxPool2D(),
    Conv2D(256, (4,4),  kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)),
    ReLU(),
    Flatten(),
    Dense(4096, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias)

                   ])
    # model = Sequential([
    # Conv2D(128, (2,2), input_shape=input_shape,
    #                kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)),
    # BatchNormalization(),
    # ReLU(),
    # Conv2D(64, (2,2), input_shape=input_shape,
    #                kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)),
    # BatchNormalization(),
    # ReLU(),
    # Conv2D(128, (2,2), input_shape=input_shape,
    #                kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)),
    # BatchNormalization(),
    # ReLU(),
    #
    # Conv2D(128, (5,5),
    #                  kernel_initializer=initialize_weights,
    #                  bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)),
    # BatchNormalization(),
    # ReLU(),
    # MaxPool2D(pool_size=(3,3)),
    # Conv2D(256, (5,5), kernel_initializer=initialize_weights,
    #                  bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)),
    # ReLU(),
    # MaxPool2D(pool_size=(5,5)),
    # Flatten(),
    #
    # Dense(2048, activation='sigmoid',
    #                kernel_regularizer=l2(1e-3),
    #                kernel_initializer=initialize_weights,bias_initializer=initialize_bias)
    #
    #                ])
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:tf.math.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    # return the model
    return siamese_net


def get_batch(batch_size,s="train"):
    """Create batch of n pairs, half same class, half different class"""
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples, h, w = X.shape

    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes,size=(batch_size,),replace=False)

    # initialize 2 empty arrays for the input image batch
    pairs=[np.zeros((batch_size, h, w,1)) for i in range(2)]

    # initialize vector for the targets
    targets=np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size//2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i,:,:,:] = X[category, idx_1].reshape(h, w, 1)
        idx_2 = rng.randint(0, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1,n_classes)) % n_classes

        pairs[1][i,:,:,:] = X[category_2,idx_2].reshape(h, w,1)

    return pairs, targets


def generate(batch_size, s="train"):
    """a generator for batches, so model.fit_generator can be used. """
    while True:
        pairs, targets = get_batch(batch_size,s)
        yield (pairs, targets)

def make_oneshot_task(N, s="val", language=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes
    n_classes, n_examples,h, w = X.shape

    indices = rng.randint(0, n_examples,size=(N,))
    if language is not None: # if language is specified, select characters for that language
        low, high = categories[language]
        if N > high - low:
            raise ValueError("This language ({}) has less than {} letters".format(language, N))
        categories = rng.choice(range(low,high),size=(N,),replace=False)

    else: # if no language specified just pick a bunch of random letters
        categories = rng.choice(range(n_classes),size=(N,),replace=False)
    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples,replace=False,size=(2,))
    test_image = np.asarray([X[true_category,ex1,:,:]]*N).reshape(N, h,w,1)
    support_set = X[categories,indices,:,:]
    support_set[0,:,:] = X[true_category,ex2]
    support_set = support_set.reshape(N, h, w,1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image,support_set]

    return pairs, targets



def test_oneshot(model, N, k, s = "val", verbose = 0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k,N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N,s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct+=1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct,N))
    return percent_correct





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

    dataset_dir = FLAGS.data_dir
    batch_size=FLAGS.batch_size
    # epochs=FLAGS.epochs
    # mode=FLAGS.mode
    train_dir = os.path.join(dataset_dir, 'train')
    validation_dir = os.path.join(dataset_dir, 'eval')
    test_dir = os.path.join(dataset_dir, 'test')
    # classes= os.listdir(train_dir)
    model = get_siamese_model((220, 120, 1))
    model.summary()
    # optimizer = Adam(learning_rate=0.001)
    optimizer = Adam(learning_rate=0.00006)
    model.compile(loss="binary_crossentropy",optimizer=optimizer)
    X,y,c = loadimgs(train_dir)
    with open(os.path.join(dataset_dir,"train.pickle"), "wb") as f:
        pickle.dump((X,c),f)


    Xval,yval,cval=loadimgs(validation_dir)
    with open(os.path.join(dataset_dir,"val.pickle"), "wb") as f:
        pickle.dump((Xval,cval),f)


    with open(os.path.join(dataset_dir, "train.pickle"), "rb") as f:
        (Xtrain, train_classes) = pickle.load(f)

    with open(os.path.join(dataset_dir, "val.pickle"), "rb") as f:
        (Xval, val_classes) = pickle.load(f)

    evaluate_every = 500 # interval for evaluating on one-shot tasks
    n_iter = 7500 # No. of training iterations
    N_way = 18 # how many classes for testing one-shot tasks
    n_val = 100 # how many one-shot tasks to validate on
    best = -1

    print("Starting training process!")
    print("-------------------------------------")
    t_start = time.time()

    # history = model.fit(generate(4, "train"), steps_per_epoch=10, epochs=1, validation_data=generate(4, "validate"))
    # print(history)

    for i in range(1, n_iter+1):
        (inputs,targets) = get_batch(batch_size)
        loss = model.train_on_batch(inputs, targets)
        if i % evaluate_every == 0:
            print("\n ------------- \n")
            print("Time for {0} iterations: {1} mins".format(i, (time.time()-t_start)/60.0))
            print("Train Loss: {0}".format(loss))
            val_acc = test_oneshot(model, N_way, n_val, verbose=True)
            model.save_weights(os.path.join(dataset_dir, 'weights.{}.h5'.format(i)))
            if val_acc >= best:
                print("Current best: {0}, previous best: {1}".format(val_acc, best))
                best = val_acc
