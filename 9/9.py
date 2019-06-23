from os import listdir
import numpy as np
import skimage
# import KMeans
# from PIL import Image

# 1a
# K = 4, random initializations of the centers, Repeat k-means clustering 10 times.
# For each run, report on the clustering error.
# pick the best clustering. Display the center faces
images_tmp = []
for filename in listdir('yalefaces_cropBackground'):
    images_tmp.append(skimage.data.imread('yalefaces_cropBackground/' + filename))
    D = np.reshape(images_tmp, (len(images_tmp), -1))

K = 4  # number of clusters
cluster_mean = np.random.rand(K)
data_class = np.zeros[136]

data_class = np.argmin

"""
n_init=10 to run 10 times with different centroid seeds.
        The final results will be the best output of n_init consecutive runs in terms of inertia.
verbose=1 to print inertia, which is the Sum of squared distances of samples to their closest cluster center.
"""
kmeans = KMeans(n_clusters=9, random_state=None, verbose=1, n_init=10).fit(X)
print('kmeans:', kmeans)
print('kmeans.labels_', kmeans.labels_)
