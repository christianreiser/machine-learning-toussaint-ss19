import tensorflow as tf
import numpy as np


"""
Ex 1 a)
- run
- tensorboard --logdir ./
"""


def main():
    data = np.loadtxt(open("data2Class_adjusted.txt", "r"), dtype=np.float32)  # read data from previous exercise
    num_samples = len(data[:, 0])  # number of samples in dataset

    X = tf.placeholder(tf.float32, shape=(num_samples, 2), name='X')  # input from dataset
    p = tf.placeholder(tf.float32, shape=num_samples, name='p')  # model output (logits)
    y = tf.placeholder(tf.float32, shape=num_samples, name='labels')  # labels
    beta = tf.placeholder(tf.float32, shape=num_samples, name='beta')  # features
    # p = tf.sigmoid(  # TODO: compute p when features beta are available
    #    tf.math.multiply(
    #        tf.transpose(X), beta)
    #    )

    total_loss = loss(y, p, num_samples)  # returns total loss form labels and logits

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)  # write graph event file for TensorBoard visualization
        print(
            sess.run(
                total_loss,
                feed_dict={
                    y: data[:, 3],  # feed label data into placeholder
                    p: np.random.rand(200)}))  # TODO: feed X instead of p when model exists
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


main()
