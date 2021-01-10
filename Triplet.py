

import argparse
import sys, os

import imageio
import tensorflow as tf
import Classification_BatchDataset
import TensorflowUtils as utils
import pickle
import time

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Input, Lambda, Layer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve,roc_auc_score
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
            if len(dirlist)>1:
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


def build_network(input_shape, embeddingsize):
    '''
    Define the neural network to learn image similarity
    Input :
            input_shape : shape of input images
            embeddingsize : vectorsize used to encode our picture
    '''

    initialize_weights = tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)
    initialize_bias = tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.01)
     # Convolutional Neural Network
    network = Sequential([
    Conv2D(128, (7,7), activation='relu',
                     input_shape=input_shape,
                     kernel_initializer='he_uniform',
                     kernel_regularizer=l2(2e-4)),
    MaxPool2D(),
    Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',
                     kernel_regularizer=l2(2e-4)),
    MaxPool2D(),
    Conv2D(256, (3,3), activation='relu', kernel_initializer='he_uniform',
                     kernel_regularizer=l2(2e-4)),
                     MaxPool2D(),
    Flatten(),
    Dense(4096, activation='relu',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer='he_uniform'),


    Dense(embeddingsize, activation=None,
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer='he_uniform'),

    #Force the encoding to live on the d-dimentional hypershpere
    Lambda(lambda x: tf.math.l2_normalize(x,axis=-1))
    ])

    return network



class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = tf.math.reduce_sum(tf.math.square(anchor-positive), axis=-1)
        n_dist = tf.math.reduce_sum(tf.math.square(anchor-negative), axis=-1)
        return tf.math.reduce_sum(tf.math.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss


def build_model(input_shape, network, margin=0.2):
    '''
    Define the Keras Model for training
        Input :
            input_shape : shape of input images
            network : Neural network to train outputing embeddings
            margin : minimal distance between Anchor-Positive and Anchor-Negative for the lossfunction (alpha)

    '''
     # Define the tensors for the three input images
    anchor_input = Input(input_shape, name="anchor_input")
    positive_input = Input(input_shape, name="positive_input")
    negative_input = Input(input_shape, name="negative_input")

    # Generate the encodings (feature vectors) for the three images
    encoded_a = network(anchor_input)
    encoded_p = network(positive_input)
    encoded_n = network(negative_input)

    #TripletLoss Layer
    loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])

    # Connect the inputs with the outputs
    network_train = Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)

    # return the model
    return network_train

def compute_dist(a,b):
    return np.sum(np.square(a-b))



def get_batch_random(batch_size,s="train"):
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
    triplets=[np.zeros((batch_size, h, w,1)) for i in range(3)]

    # initialize vector for the targets
    #targets=np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    #targets[batch_size//2:] = 1
    for i in range(batch_size):
        anchor_class = np.random.randint(0, n_classes)
        num_samples_for_class = X[anchor_class].shape[0]

        [idx_A, idx_P] = np.random.choice(num_samples_for_class, size=2, replace=False)
        negative_class = (anchor_class+np.random.randint(1, n_classes))%n_classes
        num_samples_for_n_class = X[negative_class].shape[0]
        idx_N = np.random.randint(0, num_samples_for_n_class)


        triplets[0][i,:,:,:] =  X[anchor_class,idx_A].reshape(h,w,1)
        triplets[1][i,:,:,:] =  X[anchor_class,idx_P].reshape(h,w,1)
        triplets[2][i,:,:,:] =  X[anchor_class,idx_N].reshape(h,w,1)



    return triplets


def get_batch_hard(draw_batch_size,hard_batchs_size,norm_batchs_size,network,s="train"):
    """
    Create batch of APN "hard" triplets

    Arguments:
    draw_batch_size -- integer : number of initial randomly taken samples
    hard_batchs_size -- interger : select the number of hardest samples to keep
    norm_batchs_size -- interger : number of random samples to add

    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """
    if s == 'train':
        X = Xtrain
    else:
        X = Xval

    m, w, h= X[0].shape


    #Step 1 : pick a random batch to study
    studybatch = get_batch_random(draw_batch_size,s)

    #Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
    studybatchloss = np.zeros((draw_batch_size))
    print(studybatch[0].shape)
    #Compute embeddings for anchors, positive and negatives
    A = network.predict(studybatch[0])
    P = network.predict(studybatch[1])
    N = network.predict(studybatch[2])

    #Compute d(A,P)-d(A,N)
    studybatchloss = np.sum(np.square(A-P),axis=1) - np.sum(np.square(A-N),axis=1)
    #Sort by distance (high distance first) and take the
    selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]
    #Draw other random samples from the batch
    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size),selection),norm_batchs_size,replace=False)
    selection = np.append(selection,selection2)
    triplets = [studybatch[0][selection,:,:,:], studybatch[1][selection,:,:,:], studybatch[2][selection,:,:,:]]

    return triplets

def compute_probs(network,X,Y):
    '''
    Input
        network : current NN to compute embeddings
        X : tensor of shape (m,w,h,1) containing pics to evaluate
        Y : tensor of shape (m,) containing true class

    Returns
        probs : array of shape (m,m) containing distances

    '''
    c, n, h, w = X.shape
    m = c*n
    Xa = X.reshape(m, h,w)
    Xr = Xa.reshape(m,h,w,1)
    nbevaluation = int(m*(m-1)/2)
    probs = np.zeros((nbevaluation))
    y = np.zeros((nbevaluation))

    embeddings = network.predict(Xr)

    size_embedding = embeddings.shape[1]
    k = 0
    for i in range(m):
            #Against all other images
            for j in range(i+1,m):
                #compute the probability of being the right decision : it should be 1 for right class, 0 for all other classes
                probs[k] = -compute_dist(embeddings[i,:],embeddings[j,:])

                if (Y[i]==Y[j]):
                    y[k] = 1
                    #print("{3}:{0} vs {1} : {2}\tSAME".format(i,j,probs[k],k))
                else:
                    y[k] = 0
                    #print("{3}:{0} vs {1} : \t\t\t{2}\tDIFF".format(i,j,probs[k],k))
                k += 1

    return probs,y

#probs,yprobs = compute_probs(network,x_test_origin[:10,:,:,:],y_test_origin[:10])

def compute_metrics(probs,yprobs):
    '''
    Returns
        fpr : Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i]
        tpr : Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
        thresholds : Decreasing thresholds on the decision function used to compute fpr and tpr. thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        auc : Area Under the ROC Curve metric
    '''
    # calculate AUC
    auc = roc_auc_score(yprobs, probs)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(yprobs, probs)

    return fpr, tpr, thresholds,auc


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

        probs = model.predict(inputs[0])
        probs_test = model.predict(inputs[1])
        par = np.zeros(N)

        for i in range(N):
            par[i] = compute_dist(probs[i], probs_test[i])
            print(par[i])

        if np.argmax(par) == np.argmax(targets):
            n_correct+=1
            print("T got index {} and  answer is {}".format(np.argmax(par),np.argmax(targets) ))

        else:
            print("F got index {} while answer is {}".format(np.argmax(par),np.argmax(targets) ))
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
    parser.add_argument('-n', default=100, type=int)

    FLAGS, unparsed = parser.parse_known_args()

    dataset_dir = FLAGS.data_dir
    batch_size=FLAGS.batch_size
    nperclass = FLAGS.n
    # epochs=FLAGS.epochs
    # mode=FLAGS.mode
    train_dir = os.path.join(dataset_dir, 'train')
    validation_dir = os.path.join(dataset_dir, 'validate')
    test_dir = os.path.join(dataset_dir, 'test')
    # classes= os.listdir(train_dir)

    #####################################################################

    network = build_network((220, 120, 1), embeddingsize=64)
    network_train = build_model((220, 120, 1),network)
    optimizer = Adam(learning_rate=0.00006)
    network_train.compile(optimizer=optimizer)
    network_train.summary()

    ######################################################################

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


    # evaluate_every = 1 # interval for evaluating on one-shot tasks
    # n_iter = 7500 # No. of training iterations
    # N_way = 18 # how many classes for testing one-shot tasks
    # n_val = 100 # how many one-shot tasks to validate on
    # best = -1
    #
    # print("Starting training process!")
    # print("-------------------------------------")
    # t_start = time.time()

    # history = model.fit(generate(4, "train"), steps_per_epoch=10, epochs=1, validation_data=generate(4, "validate"))
    # print(history)

    for i in range(1, 32000):
        triplets = get_batch_hard(16, 4,4, network )
        #print(triplets[0].shape)
        loss = network_train.train_on_batch(triplets, None)
        if i%100==0:
            print("Loss is {}".format(loss))
        if i%500 == 0:

            a = test_oneshot(network, 18, 100, verbose=True)
            print(a)
