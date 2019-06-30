# Exercise 10
import numpy as np
from tensorflow import keras
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf


data = np.loadtxt('data2ClassHastie.txt', dtype=np.float32)
X = data[:, 0:2]
# X = X.reshape(-1, 1)
y = data[:, 2]
print(X)
# num_samples = len(data[:, 0])  # number of samples in dataset
# print(data)
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)
predictions = neigh.predict(X)
error_knn = sum(abs(predictions-y))

print('predictions', predictions)
print('error_knn', np.sum(error_knn))


model = keras.Sequential([
    keras.layers.Dense(3),
    keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=36)

test_loss, test_acc = model.evaluate(X, y)

print('Test accuracy:', test_acc)

