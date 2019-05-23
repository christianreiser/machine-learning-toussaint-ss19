import tensorflow as tf
import numpy as np


"""
Ex 1 a)
- run
- tensorboard --logdir ./
"""

def main():
    data = np.loadtxt(open("data2Class_adjusted.txt", "r"), dtype=np.float32)
    num_samples = len(data[:, 0])

    X = tf.placeholder(tf.float32, shape=(num_samples, 2), name='X')
    p = tf.placeholder(tf.float32, shape=num_samples, name='p')  # TODO: remove after model returns p
    y = tf.placeholder(tf.float32, shape=num_samples, name='labels')
    beta = tf.placeholder(tf.float32, shape=num_samples, name='beta')
    # p = tf.sigmoid(  # TODO: compute p when features beta are available
    #    tf.math.multiply(
    #        tf.transpose(X), beta)
    #    )

    total_loss = loss(y, p, num_samples)
    print(total_loss.shape)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        print(
            sess.run(
                total_loss,
                feed_dict={
                    y: data[:, 3],
                    p: np.random.rand(200)}))  # TODO: feed X instead of p when model exists
    writer.close()


def loss(labels, logits, num_samples):
    """
    Computes the total loss for logistic regression
    """
    ones_vec = np.ones(num_samples, dtype=np.float32)
    loss_vec = tf.math.add(
        tf.multiply(labels, tf.math.log(logits)),
        tf.multiply(
            tf.math.subtract(ones_vec, labels),
            tf.math.log(
                tf.math.subtract(ones_vec, logits)
            )
        )
    )
    total_loss = -tf.reduce_sum(loss_vec)
    return total_loss


main()
