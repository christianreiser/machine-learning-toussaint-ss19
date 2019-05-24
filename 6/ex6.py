import tensorflow as tf
import numpy as np


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
    X = tf.placeholder(tf.float32, shape=(num_samples, 3), name='X')  # input from dataset
    y = tf.placeholder(tf.float32, shape=num_samples, name='labels')  # labels
    beta = tf.placeholder(tf.float32, shape=num_samples, name='beta')  # features

    p = predict(X, beta)  # compute prediction p from input data and features beta
    total_loss = loss(y, p, num_samples)  # returns total loss form labels and logits
    gradients = tf.gradients(total_loss, beta, name='gradients')  # compute gradients of loss w.r.t. beta
    hessians = tf.hessians(total_loss, beta, name='hessians')  # compute hessians of loss w.r.t. beta
    # L, dL, ddL = numpy_equations(X, beta, y)  # TODO: fix bug
    # print('L:', L, 'dL:', dL, 'ddL', ddL)  # TODO: compare to tf results

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)  # write graph event file for TensorBoard visualization
        print(
            sess.run(
                (total_loss, gradients, hessians),  # TODO: ex a: total_loss
                feed_dict={
                    X: data[:, 0:3],  # feed input data X from dataset
                    y: data[:, 3],  # feed label data into placeholder
                    beta: np.random.rand(200)  # TODO: feed beta from model
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
    L = -np.sum(y * np.log(p) + ((1. - y) * np.log(1.-p)))
    dL = np.dot(X.T, p - y)
    W = np.identity(X.shape[0]) * p * (1. - p)
    ddL = np.dot(X.T, np.dot(W, X))
    return L, dL, ddL


"""
exercise 2
"""


def exercise2():
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
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)  # sigmoid
    mean_cross_entropy_loss = tf.reduce_mean(cross_entropy)  # mean of loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)  # set up Adam optimizer
    training_operation = optimizer.minimize(mean_cross_entropy_loss)  # optimize nn parameters to minimize loss mean

    # Session
    with tf.Session() as sess:
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


def nn(x):
    """
    Two Layer NN
    :param x: input data sample
    :param mu: mean of gaussian initialization distribution
    :param sigma: variance of initialization norm distribution
    :return: logits
    """
    # Layer 1: Fully Connected. Input = 3x1. Output = 100x1.
    fc1_W = tf.Variable(tf.random.truncated_normal(shape=(3, 100), mean=0, stddev=1/np.sqrt(3)))  # colocate_is with out dated
    fc1_b = tf.Variable(tf.zeros(1))
    fc1 = tf.add(tf.matmul(x, fc1_W), fc1_b)
    fc1 = tf.nn.leaky_relu(fc1)  # Activation.
    # fc1 = tf.nn.dropout(fc1, dropout)

    # Layer 2: Fully Connected. Input = 100. Output = 1.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(100, 1), mean=0, stddev=1/np.sqrt(100)))
    fc2_b = tf.Variable(tf.zeros(1))
    logits = tf.add(tf.matmul(fc1, fc2_W), fc2_b)
    return logits


exercise1()
exercise2()
