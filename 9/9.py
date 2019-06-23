# TODO: my has same dim as image


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

# initialize
num_clusters = 4  # number of clusters
cluster_mean = np.random.randint(low=0, high=255, size=num_clusters)  # ini cluster mean randomly
print('initial cluster mean', cluster_mean)
# data_class = np.zeros(num_imgs)
distances = np.zeros([num_imgs, len_flat_img, num_clusters])  # ini centroid data distance tensor
distances_sum = np.zeros([num_imgs, num_clusters])
# distances in first column and cluster index in second (c in sheet)
distance_sum_nearest_cluster = np.zeros([num_imgs, 2])

# assign images to clusters
for cluster in range(num_clusters):
    distances[:, :, cluster] = np.subtract(D, cluster_mean[cluster])  # subtract cluster mean from evey pixel
# distances[:, :, cluster] = np.power(distances[:, :, cluster], 2)  # TODO: power on, careful for mean distance
    for image in range(num_imgs):
        distances_sum[image, cluster] = np.sum(distances[image, :, cluster])  # sum distances
for image in range(num_imgs):
    nearest_cluster_index = np.argmin(distances_sum[image, :])  # nearest cluster index
    distance_sum_nearest_cluster[image, 1] = nearest_cluster_index  # store nearest cluster index
    distance_sum_nearest_cluster[image, 0] = distances_sum[image, nearest_cluster_index]  # nearest distance
print('distance_sum_nearest_cluster[:, 0]', distance_sum_nearest_cluster[:, 0])

# sum up distances of clusters
distances_assigned_to_cluster = np.zeros([num_clusters, num_imgs])
for image in range(num_imgs):
    if distance_sum_nearest_cluster[image, 1] == 0:
        distances_assigned_to_cluster[0, image] = distance_sum_nearest_cluster[image, 0]
    elif distance_sum_nearest_cluster[image, 1] == 1:
        distances_assigned_to_cluster[1, image] = distance_sum_nearest_cluster[image, 0]
    elif distance_sum_nearest_cluster[image, 1] == 2:
        distances_assigned_to_cluster[2, image] = distance_sum_nearest_cluster[image, 0]
    elif distance_sum_nearest_cluster[image, 1] == 3:
        distances_assigned_to_cluster[3, image] = distance_sum_nearest_cluster[image, 0]
    else:
        print('WARNING: falsely assigned!')


# relocate cluster mean
for cluster in range(num_clusters):
    if len(distance_sum_nearest_cluster[cluster]) > 0:
        cluster_mean[cluster] = np.mean(distances_assigned_to_cluster[cluster])
    else:
        cluster_mean[cluster] = np.random.randint(low=0, high=255, size=num_clusters)  # ini cluster mean randomly
        print('Empty cluster', cluster, 'relocated randomly.')

print('assigned clusters of images:', distance_sum_nearest_cluster[:, 1])
print('cluster mean=', cluster_mean)
"""
n_init=10 to run 10 times with different centroid seeds.
        The final results will be the best output of n_init consecutive runs in terms of inertia.
verbose=1 to print inertia, which is the Sum of squared distances of samples to their closest cluster center.
"""
# kmeans = KMeans(n_clusters=9, random_state=None, verbose=1, n_init=10).fit(X)
# print('kmeans:', kmeans)
# print('kmeans.labels_', kmeans.labels_)
