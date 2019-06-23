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
print('D.shape', D.shape)
num_imgs = D.shape[0]
len_flat_img = D.shape[1]


num_clusters = 4  # number of clusters
cluster_mean = np.random.rand(num_clusters)
data_class = np.zeros(num_imgs)
distances = np.zeros([num_imgs, len_flat_img, num_clusters])  # ini centroid data distance tensor
distances_sum = np.zeros([num_imgs, num_clusters])
distance_nearest_cluster = np.zeros([num_imgs, 2])
for cluster in range(num_clusters):
    distances[:, :, cluster] = np.subtract(D, cluster_mean[cluster])
    distances[:, :, cluster] = np.power(distances[:, :, cluster], 2)
    for image in range(num_imgs):
        distances_sum[image, cluster] = np.sum(distances[image, :, cluster])
for image in range(num_imgs):
    nearest_cluster_indice = np.argmin(distances_sum[image, :])  # nearest cluster indice
    distance_nearest_cluster[image, 1]
    distance_nearest_cluster[image, 0] = distances_sum[image, distance_nearest_cluster[image, 1]]  # nearest distance



data_class[i] = np.argmin

"""
n_init=10 to run 10 times with different centroid seeds.
        The final results will be the best output of n_init consecutive runs in terms of inertia.
verbose=1 to print inertia, which is the Sum of squared distances of samples to their closest cluster center.
"""
# kmeans = KMeans(n_clusters=9, random_state=None, verbose=1, n_init=10).fit(X)
# print('kmeans:', kmeans)
# print('kmeans.labels_', kmeans.labels_)
