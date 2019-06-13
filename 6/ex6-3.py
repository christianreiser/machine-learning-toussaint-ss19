from random import randint
import numpy as np
import tensorflow as tf
import random


x = np.ones(1000)
K = np.random.randint(low=0, high=100000, size=32)
J = np.zeros(shape=(100000, 1000))
for row in range(1**3):
    for element in range(9):
        column = np.random.randint(0, 999)
        rdm_bool = random.choice([True, False])
        if rdm_bool == True:
            J[row, column] = np.random.normal(loc=0.0, scale=1.0, size=None)  # random sample from normal dist with sigma = 1
        else:
            np.random.normal(loc=0.0, scale=100.0, size=None)  # random sample from normal dist with sigma = 100
print(J)

g = tf.multiply(
    tf.math.divide(1, K),
    tf.reduce_sum(
        tf.multiply(
            np.transpose(J),
            tf.multiply(J, x)
        )
    )
)

l = tf.divide(1,
              tf.multiply(2, n))
l = tf.multiply(l, tf.transpose(x))
l = tf.multiply(l, H)
l = tf.multiply(l, x)
l
print(K)
