import tensorflow as tf


X = tf.placeholder(tf.float32, shape=(100, 3))
y = tf.placeholder(tf.float32, shape=(100, 1))
beta = tf.placeholder(tf.float32, shape=3)

p = 1


def loss(
        labels=y,
        logits=p):
    """
    # TODO: describe function
    Computes sigmoid cross entropy given `logits`.
    """
    ones_vec = tf.ones([tf.size(labels)[0], 1], tf.int32)

    loss_vec = tf.math.add(
        labels * tf.math.log(logits),
        tf.math.subtract(ones_vec - labels) * tf.math.log(tf.math.subtract(ones_vec, logits))
    )

    total_loss = -tf.math.add_n(loss_vec)
    return total_loss


total_loss = loss(y, p)

print(total_loss)
