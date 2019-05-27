from random import randint
import numpy as np
import tensorflow as tf


x = np.ones(1000)
K = np.random.randint(low=0, high=100000, size=32)
J = scipy.sparse.random(10**5, 1**3)


g = tf.multiply(
    tf.math.divide(1, K),
    tf.reduce_sum(
        tf.multiply(
            np.transpose(J),
            tf.multiply(J, x)
        )
    )
)


print(K)
