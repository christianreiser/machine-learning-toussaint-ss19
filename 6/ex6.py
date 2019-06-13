import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import skimage
from skimage import transform
from skimage.color import rgb2gray
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt


def exercise1():
    """
    Ex 1 a)
    - placeholders, session, feed, loss
    - TensorBoard --logdir ./
    """
    # read data
    data = np.loadtxt('data2Class_adjusted.txt', dtype=np.float32)  # read data from previous exercise
    num_samples = len(data[:, 0])  # number of samples in dataset

    # Initialize tf.placeholders
    X = tf.placeholder(tf.float32, shape=(num_samples,  3), name='X')  # input from dataset
    y = tf.placeholder(tf.float32, shape=num_samples, name='labels')  # labels
    beta = tf.placeholder(tf.float32, shape=(3, 1), name='beta')  # features

    beta_feed = np.random.rand(3, 1)  # same beta for feed and numpy equations
    p = predict(X, beta)  # compute prediction p from input data and features beta
    total_loss = loss(y, p, num_samples)  # returns total loss form labels and logits

    # gradients hessians
    gradients = tf.gradients(total_loss, beta, name='gradients')  # compute gradients of loss w.r.t. beta
    hessians = tf.hessians(total_loss, beta, name='hessians')  # compute hessians of loss w.r.t. beta

    # numpy equations
    L, dL, ddL = numpy_equations(data[:, 0:3], beta_feed, data[:, 3])  # np equations from ex sheet
    print('L:', L, 'dL:', dL, 'ddL', ddL)  # TODO: compare to tf results

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)  # write graph event file for TensorBoard visualization
        print(
            sess.run(
                (total_loss, gradients, hessians),  # TODO: ex a: compare to np equations
                feed_dict={
                    X: data[:, 0:3],  # feed input data X from dataset
                    y: data[:, 3],  # feed label data into placeholder
                    beta: beta_feed  # TODO: feed beta from model?
                }
            )
        )
        writer.close()


def loss(labels, logits, num_samples):
    """
    Computes the total loss for logistic regression
    L(beta) = -sum[ y*log(p) + (1-y)*log(1-p) ]
    """
    ones_vec = np.ones(num_samples, dtype=np.float32) # vector with ones
    loss_vec = tf.math.add(  # loss vector, each entry corresponds to one data sample
        tf.multiply(labels, tf.math.log(logits)),
        tf.multiply(
            tf.math.subtract(ones_vec, labels),
            tf.math.log(
                tf.math.subtract(ones_vec, logits)
            )
        )
    )
    total_loss = -tf.reduce_sum(loss_vec)  # reduce vector by summing up
    return total_loss


def predict(X, beta):
    """
    computes predictions p from input data and features
    p = sigmoid( transp(X) * beta)
    """
    p = tf.sigmoid(
        tf.math.multiply(
            tf.transpose(X), beta)
        )
    return p


def numpy_equations(X, beta, y):
    p = 1. / (1. + np.exp(-np.dot(X, beta)))
    L = -np.sum(y * np.log(p) + ((1. - y) * np.log(1. - p)))
    dL = np.dot(X.T, p - y)
    W = np.identity(X.shape[0]) * p * (1. - p)
    ddL = np.dot(X.T, np.dot(W, X))
    return L, dL, ddL


"""
/////////////////// exercise 2 /////////////////////////////////////////////////////////////////////////
"""


def exercise2a():
    # hyper-parameters
    num_epochs = 100  # number of training epochs
    lr = 0.001  # learning rate

    # read data
    data = np.loadtxt('data2Class_adjusted.txt', dtype=np.float32)  # read data from previous exercise

    # Initialize tf.placeholders
    X = tf.placeholder(tf.float32, shape=(None, 3), name='X')  # input from dataset
    y = tf.placeholder(tf.float32, shape=(None, 1), name='labels')  # labels

    # build tf graph
    logits = nn(X)  # neural net computes logits from input data
    hinge_loss = tf.losses.hinge_loss(labels=y, logits=logits)  # sigmoid
    mean_hinge_loss = tf.reduce_mean(hinge_loss)  # mean of loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # set up Adam optimizer
    training_operation = optimizer.minimize(mean_hinge_loss)  # optimize nn parameters to minimize loss mean

    # Session
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)  # write graph event file for TensorBoard visualization
        sess.run(tf.global_variables_initializer())  # initialize variables
        for epoch in range(num_epochs):  # iterate for number of epochs
            batch_x, batch_y = data[:, 0:3], data[:, 3:4]  # batch is whole dataset
            sess.run(
                (
                    # print('mean loss in epoch', epoch, '=', mean_cross_entropy_loss),
                    training_operation
                ),
                feed_dict={
                    X: batch_x,  # feed input data X from dataset
                    y: batch_y,  # feed label data into placeholder
                }
            )
        writer.close()  # close event file


def nn(x):
    """
    Two Layer NN
    :param x: input data sample
    :param mu: mean of gaussian initialization distribution
    :param sigma: variance of initialization norm distribution
    :return: logits
    """
    # Layer 1: Fully Connected. Input = 3x1. Output = 100x1.
    fc1_W = tf.Variable(tf.random.truncated_normal(shape=(3, 100), mean=0, stddev=1/np.sqrt(3)))  # Initialize weights TODO: colocate_is with out dated
    fc1_b = tf.Variable(
        tf.zeros(1))  #
    fc1 = tf.add(tf.matmul(x, fc1_W), fc1_b)
    fc1 = tf.nn.leaky_relu(fc1)  # Activation.
    # fc1 = tf.nn.dropout(fc1, dropout)

    # Layer 2: Fully Connected. Input = 100. Output = 1.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(100, 1), mean=0, stddev=1/np.sqrt(100)))
    fc2_b = tf.Variable(tf.zeros(1))
    logits = tf.add(tf.matmul(fc1, fc2_W), fc2_b)
    return logits


"""
/////////////////// exercise 2b /////////////////////////////////////////////////////////////////////////
"""


def load_data(data_directory):
    """
    modified from exercise sheet to output 4dim array
    :param data_directory:
    :return: 4 dim array of 3d images
    """
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    tmp_images = []
    # images = np.ndarray(shape=(4000, 128, 128))
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            # images = np.concatenate(skimage.data.imread(f), axis=0)
            tmp_images.append(
                skimage.transform.resize(
                    skimage.data.imread(f), output_shape=(128, 128, 3))
            )
            labels.append(int(d))
    images = np.concatenate([image[np.newaxis] for image in tmp_images])

    return np.array(images), np.array(labels)



def plot_data(signs, labels):
    for i in range(len(signs)):
        plt.subplot(4, len(signs)/4 + 1, i+1)
        plt.axis('off')
        plt.title("Label {0}".format(labels[i]))


def exercise2b():
    train_images, train_labels = load_data(data_directory='./BelgiumTSC/Training')  # import data
    test_images, test_labels = load_data(data_directory='./BelgiumTSC/Testing')  # import data

    """
    # plot one image
    print(images[0].shape)
    plt.figure()
    plt.imshow(images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()

    # plot 10 images
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.show()
    """

    """
    # create model
    model = tf.keras.Sequential()
    # add model layers
    model.add(layers.Conv3D(filters=64, kernel_size=3, activation='relu', input_shape=(128, 128, 3, 1)))  # TODO: leaky
    model.add(layers.Conv3D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.Conv3D(filters=16, kernel_size=3, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=62,
        activation='softmax',
        use_bias=True,=
        kernel_initializer=tf.initializers.he_normal))

    # plot_data(signs=, labels=)
    """

    model = keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(128, 128, 3)),  # TODO: leaky
        keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
        keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),
        keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),
        keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'),
        keras.layers.Flatten(input_shape=(64, 64, 3)),
        keras.layers.Dense(62, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=1)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)


# exercise1()
# exercise2a()
exercise2b()

