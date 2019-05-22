import tensorflow as tf
import numpy as np

def main():
    X = tf.placeholder(tf.float32, shape=(100, 3), name='X')
    y = tf.placeholder(tf.float32, shape=(100, 1), name='labels')
    p = tf.placeholder(tf.float32, shape=(100, 1), name='logits')
    beta = tf.placeholder(tf.float32, shape=(3, 1), name='beta')

    total_loss = loss(y, p)

    with tf.Session() as sess:
        print(sess.run(total_loss, feed_dict={y: np.random.rand(100, 1), p: np.random.rand(100, 1)}))


def loss(labels, logits):
    """
    Computes the total loss for logistic regression
    """
    ones_vec = np.ones((100, 1), dtype=np.float32)
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

